"""The URDF reader: the tree it builds, the hash that pins the file, and the three groups every element falls into,
with one warning per kind of element not used and a refusal naming each refused one."""

import hashlib
import math
import re
import textwrap
import warnings

import numpy as np
import pytest

import multibody_cases as cases
from mime.core.multibody import CATEGORY_MESSAGES, DescriptionRefused, DescriptionWarning, ElementGroup, \
    InvalidDescription, JointKind, read_urdf

READ, UNUSED, REFUSED = ElementGroup.READ, ElementGroup.UNUSED, ElementGroup.REFUSED


def urdf(tmp_path, body: str, name: str = "robot.urdf"):
    path = tmp_path / name
    path.write_text(f'<?xml version="1.0"?>\n<robot name="r">\n{textwrap.dedent(body)}\n</robot>\n')
    return path


ARM = """
<link name="base"/>
<link name="arm">
  <inertial><origin xyz="0.5 0 0"/><mass value="2"/>
    <inertia ixx="0.01" iyy="0.2" izz="0.2" ixy="0" ixz="0" iyz="0"/></inertial>
</link>
<joint name="shoulder" type="revolute">
  <parent link="base"/><child link="arm"/><origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
  <limit lower="-1" upper="1" effort="10" velocity="2"/>
  <dynamics damping="0.3" friction="0.1"/>
</joint>
"""


def read_quietly(path, **options):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reading = read_urdf(path, **options)
    return reading, [w for w in caught if issubclass(w.category, DescriptionWarning)]


# The tree -----------------------------------------------------------------------------------------------------------

def test_the_planar_arm_reads_into_its_tree_with_every_element_used_and_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reading = read_urdf(cases.PLANAR_URDF)
    tree = reading.tree
    assert (tree.name, tree.root) == ("three_link_planar", "base_link")
    assert [b.name for b in tree.bodies] == ["base_link", "link_1", "link_2", "link_3"]
    assert [(j.name, j.kind, j.parent, j.child) for j in tree.joints] == [
        ("joint_1", JointKind.REVOLUTE, "base_link", "link_1"), ("joint_2", JointKind.REVOLUTE, "link_1", "link_2"),
        ("joint_3", JointKind.REVOLUTE, "link_2", "link_3")]
    assert [j.damping for j in tree.joints] == [0.05, 0.05, 0.02]
    assert tree.joint("joint_3").limits.effort == 50.0 and tree.joint("joint_1").limits.lower == -3.1416
    assert tree.body("link_2").inertial.mass == 1.5
    assert tree.body("link_2").inertial.centre_of_mass == (0.5, 0.0, 0.0)
    assert not reading.classification.group(UNUSED) and not reading.classification.group(REFUSED)
    assert {e.category for e in reading.classification.group(READ)} == {
        "link", "inertial", "origin", "mass", "inertia", "joint", "parent", "child", "axis", "limit", "dynamics"}


def test_the_ar4_reads_with_its_visual_links_welded_and_one_warning_per_kind_not_used():
    reading, caught = read_quietly(cases.AR4_URDF)
    tree, classification = reading.tree, reading.classification
    assert tree.root == "world" and len(tree.bodies) == 27 and len(tree.joints) == 26
    assert tree.articulation.dof_names == tuple(f"joint_{i}" for i in range(1, 7))
    assert tree.joint("joint_2").limits.effort == -1.0  # stored as the file gives it
    assert set(classification.categories(UNUSED)) == {"visual", "collision"}
    assert len(classification.categories(UNUSED)["visual"]) == 18
    assert len(classification.categories(UNUSED)["collision"]) == 4
    assert not classification.group(REFUSED)
    assert len(caught) == 3  # visuals, collisions, and the links read as massless
    assert [str(w.message).split(":")[1].split()[1:3] for w in caught[:2]] == [["<visual>", "elements"],
                                                                                ["<collision>", "elements"]]
    assert len(classification.inertia_gaps) == 18
    assert not any(gap.needed_by_dynamics for gap in classification.inertia_gaps)
    assert {gap.moves_with for gap in classification.inertia_gaps} == {"world", *(f"link_{i}" for i in range(1, 7))}


def test_the_tree_records_the_files_hash_and_a_pinned_reading_refuses_another_file(tmp_path):
    digest = hashlib.sha256(cases.PLANAR_URDF.read_bytes()).hexdigest()
    reading = read_urdf(cases.PLANAR_URDF, sha256=digest)
    assert reading.tree.source.sha256 == digest and reading.tree.source.format == "urdf"
    assert reading.tree.source.file_name == "three_link_planar.urdf"
    edited = tmp_path / "three_link_planar.urdf"
    edited.write_bytes(cases.PLANAR_URDF.read_bytes().replace(b'damping="0.05"', b'damping="0.06"', 1))
    with pytest.raises(InvalidDescription, match="pinned"):
        read_urdf(edited, sha256=digest)


def test_origins_turn_by_roll_then_pitch_then_yaw_about_the_fixed_axes(tmp_path):
    """URDF's rpy is R = Rz(yaw) Ry(pitch) Rx(roll), and an inertial's rpy turns its tensor into the link's frame."""
    roll, pitch, yaw = 0.3, -0.7, 1.1
    path = urdf(tmp_path, f"""
        <link name="a"/>
        <link name="b"><inertial><origin rpy="0 0 {math.pi / 2}"/><mass value="1"/>
          <inertia ixx="1" iyy="2" izz="3" ixy="0" ixz="0" iyz="0"/></inertial></link>
        <joint name="j" type="fixed"><parent link="a"/><child link="b"/>
          <origin xyz="1 2 3" rpy="{roll} {pitch} {yaw}"/></joint>""")
    tree = read_urdf(path).tree
    c, s = np.cos, np.sin
    rx = np.array([[1, 0, 0], [0, c(roll), -s(roll)], [0, s(roll), c(roll)]])
    ry = np.array([[c(pitch), 0, s(pitch)], [0, 1, 0], [-s(pitch), 0, c(pitch)]])
    rz = np.array([[c(yaw), -s(yaw), 0], [s(yaw), c(yaw), 0], [0, 0, 1]])
    assert np.allclose(tree.joint("j").origin.matrix(), rz @ ry @ rx, atol=1e-15)
    assert np.allclose(tree.body("b").inertial.inertia, np.diag([2.0, 1.0, 3.0]), atol=1e-15)


def test_a_continuous_joint_has_no_position_limits(tmp_path):
    path = urdf(tmp_path, """
        <link name="a"/><link name="b"/>
        <joint name="spin" type="continuous"><parent link="a"/><child link="b"/>
          <limit effort="1" velocity="3"/></joint>""")
    limits = read_urdf(path).tree.joint("spin").limits
    assert (limits.lower, limits.upper, limits.velocity) == (-math.inf, math.inf, 3.0)


@pytest.mark.parametrize(("body", "says"), [
    ("""<link name="a"/><link name="b"/><joint name="j" type="revolute"><parent link="a"/><child link="b"/></joint>""",
     "no <limit>"),
    ("""<link name="a"/><link name="b"/><joint name="j" type="hinge"><parent link="a"/><child link="b"/></joint>""",
     "type 'hinge'"),
    ("""<link name="a"/><link name="b"/>""", "one root link"),
    ("""<link name="a"><inertial><mass value="1"/></inertial></link>""", "both <mass> and <inertia>"),
    ("""<link name="a"><inertial><mass value="heavy"/><inertia/></inertial></link>""", "not a number"),
    ("""<link name="a"/><link name="b"/><joint name="j" type="fixed"><parent link="a"/><child link="b"/>
        <origin xyz="1 2"/></joint>""", "not 3 numbers"),
])
def test_a_malformed_file_is_refused_saying_what_is_wrong(tmp_path, body, says):
    with pytest.raises(InvalidDescription, match=says):
        read_urdf(urdf(tmp_path, body))


# The three groups ---------------------------------------------------------------------------------------------------

ONE_OF_EACH = ARM.replace('<link name="arm">', '<link name="arm">\n  <visual><geometry><box size="1 0.1 0.1"/>'
                                                '</geometry></visual>').replace(
    "</joint>", '  <mimic joint="elbow" multiplier="1"/>\n</joint>')


def test_one_element_of_each_group_gives_the_three_groups_one_warning_and_a_refusal_naming_it(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(DescriptionRefused, match=r"joint\[shoulder\]/mimic") as refused:
            read_urdf(urdf(tmp_path, ONE_OF_EACH))
    classification = refused.value.classification
    assert classification.categories(UNUSED) == {"visual": ("link[arm]/visual",)}
    assert classification.categories(REFUSED) == {"mimic": ("joint[shoulder]/mimic",)}
    assert "joint[shoulder]/limit" in classification.categories(READ)["limit"]
    assert [e.path for e in refused.value.refused] == ["joint[shoulder]/mimic"]
    assert CATEGORY_MESSAGES["mimic"] in str(refused.value) and "acknowledged=" in str(refused.value)
    warned = [str(w.message) for w in caught if issubclass(w.category, DescriptionWarning)]
    assert len(warned) == 1 and "link[arm]/visual" in warned[0] and CATEGORY_MESSAGES["visual"] in warned[0]


def test_without_the_refused_element_the_same_file_reads_and_without_the_unused_one_it_is_quiet(tmp_path):
    reading, caught = read_quietly(urdf(tmp_path, ONE_OF_EACH.replace('  <mimic joint="elbow" multiplier="1"/>\n',
                                                                      "")))
    assert reading.tree.joint("shoulder").kind is JointKind.REVOLUTE and len(caught) == 1
    reading, caught = read_quietly(urdf(tmp_path, ARM, name="plain.urdf"))
    assert reading.tree.joint("shoulder").friction == 0.1 and not caught


def test_an_acknowledged_mimic_is_read_as_an_independent_joint_and_recorded(tmp_path):
    reading, caught = read_quietly(urdf(tmp_path, ONE_OF_EACH),
                                   acknowledged={"joint[shoulder]/mimic": "the coupled finger is not simulated"})
    assert reading.tree.articulation.dof_names == ("shoulder",)
    assert reading.classification.acknowledged == (("joint[shoulder]/mimic", "the coupled finger is not simulated"),)
    assert not reading.classification.unacknowledged()
    assert len(caught) == 2 and "read past as acknowledged" in str(caught[1].message)


def test_many_elements_of_one_kind_give_one_warning_listing_them_all(tmp_path):
    visuals = "".join(f'<visual name="v{i}"><geometry><sphere radius="0.1"/></geometry></visual>' for i in range(4))
    reading, caught = read_quietly(urdf(tmp_path, ARM.replace('<link name="base"/>',
                                                              f'<link name="base">{visuals}</link>')))
    assert len(reading.classification.categories(UNUSED)["visual"]) == 4
    assert len(caught) == 2  # the visuals, and the root link read as massless
    assert all(f"link[base]/visual[v{i}]" in str(caught[0].message) for i in range(4))


@pytest.mark.parametrize("element", ["material", "transmission", "gazebo", "ros2_control"])
def test_each_recognised_extension_is_unused_and_warned_about(tmp_path, element):
    reading, caught = read_quietly(urdf(tmp_path, ARM + f'<{element} name="x"/>'))
    assert reading.classification.categories(UNUSED) == {element: (f"{element}[x]",)}
    assert len(caught) == 1 and CATEGORY_MESSAGES[element] in str(caught[0].message)


def test_a_joints_calibration_and_safety_controller_are_unused(tmp_path):
    reading, _ = read_quietly(urdf(tmp_path, ARM.replace("</joint>", '<calibration rising="0"/>'
                                                                      '<safety_controller k_velocity="1"/></joint>')))
    assert set(reading.classification.categories(UNUSED)) == {"calibration", "safety_controller"}


def test_an_element_urdf_does_not_define_is_refused_unless_acknowledged(tmp_path):
    path = urdf(tmp_path, ARM.replace("</joint>", "<stiffness value='3'/></joint>"))
    with pytest.raises(DescriptionRefused, match=r"joint\[shoulder\]/stiffness"):
        read_urdf(path)
    reading, _ = read_quietly(path, acknowledged={"joint[shoulder]/stiffness": "the joint is rigid enough"})
    assert reading.classification.categories(REFUSED) == {"unknown": ("joint[shoulder]/stiffness",)}


@pytest.mark.parametrize(("body", "path"), [
    (ARM + '<link name="free"/><joint name="float" type="floating"><parent link="arm"/><child link="free"/></joint>',
     "joint[float]"),
    (ARM + '<link name="flat"/><joint name="slab" type="planar"><parent link="arm"/><child link="flat"/></joint>',
     "joint[slab]"),
])
def test_a_joint_the_tree_cannot_hold_is_refused_and_cannot_be_acknowledged(tmp_path, body, path):
    with pytest.raises(DescriptionRefused, match=re.escape(path)):
        read_urdf(urdf(tmp_path, body))
    with pytest.raises(InvalidDescription, match="cannot be read past"):
        read_urdf(urdf(tmp_path, body), acknowledged={path: "a reason"})


def test_unexpanded_xacro_is_refused_with_how_to_expand_it(tmp_path):
    path = tmp_path / "robot.urdf.xacro"
    path.write_text('<robot name="r" xmlns:xacro="http://www.ros.org/wiki/xacro">'
                    '<xacro:property name="length" value="1"/><link name="a"/></robot>')
    with pytest.raises(DescriptionRefused, match="expand them first"):
        read_urdf(path)


def test_an_acknowledgement_naming_nothing_refused_or_giving_no_reason_is_itself_refused(tmp_path):
    with pytest.raises(InvalidDescription, match="refuses no such element"):
        read_urdf(urdf(tmp_path, ARM), acknowledged={"joint[shoulder]/mimic": "stale"})
    with pytest.raises(InvalidDescription, match="gives no reason"):
        read_urdf(urdf(tmp_path, ONE_OF_EACH), acknowledged={"joint[shoulder]/mimic": " "})


# Mass a dynamic chain would need ------------------------------------------------------------------------------------

def test_a_moving_body_with_geometry_and_no_inertial_anywhere_is_a_gap_a_dynamic_chain_needs(tmp_path):
    body = ARM + """
        <link name="hand"><visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual></link>
        <joint name="wrist" type="revolute"><parent link="arm"/><child link="hand"/>
          <limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
        <link name="cover"><visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual></link>
        <joint name="cover_mount" type="fixed"><parent link="arm"/><child link="cover"/></joint>"""
    reading, caught = read_quietly(urdf(tmp_path, body))
    gaps = {g.link: g for g in reading.classification.inertia_gaps}
    assert gaps["hand"].needed_by_dynamics and gaps["hand"].moves_with == "hand"
    assert not gaps["cover"].needed_by_dynamics and gaps["cover"].moves_with == "arm"  # welded to a body with mass
    assert "hand (needed by a dynamic chain)" in str(caught[-1].message)
    weighted = body.replace('<link name="hand">', '<link name="hand"><inertial><mass value="0.1"/>'
                                                  '<inertia ixx="1e-4" iyy="1e-4" izz="1e-4"/></inertial>')
    reading, _ = read_quietly(urdf(tmp_path, weighted, name="weighted.urdf"))
    assert [g.link for g in reading.classification.inertia_gaps] == ["cover"]
