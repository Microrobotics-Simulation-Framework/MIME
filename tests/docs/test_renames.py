"""The rename map: a ratchet that only shrinks reads a renamed entry's line as new against the base, unless
``docs/ratchets/renames.txt`` carries it over, and the map is honest. A line is carried only when the old name is
registered in the base and the new one at the head, the old name is gone from the head, and the line's reason is
unchanged; each control plants one of these failing, and a genuinely new line still fails. The base here is a
committed copy in which the entry was called ``uniform_rotating_x``, so the head (this checkout) is the renamed one."""

from checkout import copy_checkout, docs_check, edit, failures
from test_numbers import committed

LIST = "docs/ratchets/none_declared.txt"
MAP = "docs/ratchets/renames.txt"
SOURCE = "src/mime/fields/_uniform_rotating.py"
NEW, OLD = "uniform_rotating", "uniform_rotating_x"
HEAD_LINE = f"field_provider {NEW} 0.1.0 references  # present at D1b"
BASE_LINE = f"field_provider {OLD} 0.1.0 references  # present at D1b"
NOT_NEW = (f"{LIST}:10: 'field_provider {NEW} 0.1.0 references' is not on the list at HEAD, and 'present at D1b' is the "
           f"reason of the cases that stood when the list landed, which only shrinks; fix the empty field, or give the "
           f"line its own reason (logged)")


def renamed_since_the_base(tmp_path, mapped: str, head_line: str = HEAD_LINE):
    """A checkout whose base committed the entry as ``OLD`` and whose working copy has it as ``NEW`` (this checkout's),
    with ``mapped`` as the rename map."""
    root = copy_checkout(tmp_path)
    edit(root / SOURCE, f'name = "{NEW}"', f'name = "{OLD}"')
    edit(root / LIST, HEAD_LINE, BASE_LINE)
    committed(root)
    edit(root / SOURCE, f'name = "{OLD}"', f'name = "{NEW}"')
    edit(root / LIST, BASE_LINE, head_line)
    (root / MAP).write_text(mapped + "\n", encoding="utf-8")
    return root


def check(root):
    return docs_check(root, base="HEAD")


def test_a_renamed_line_is_carried_over_by_the_map_and_passes(tmp_path):
    root = renamed_since_the_base(tmp_path, f"field_provider {OLD} -> {NEW}  # 2026-10-29, #178")
    done = check(root)
    assert done.returncode == 0, done.stdout


def test_control_without_the_map_the_renamed_line_reads_as_new(tmp_path):
    root = renamed_since_the_base(tmp_path, "# an empty map")
    assert failures(check(root)) == [f"{root / LIST}:10: " + NOT_NEW.split(":10: ")[1]]


def test_control_a_map_line_whose_old_name_the_base_does_not_register_excuses_nothing(tmp_path):
    root = renamed_since_the_base(tmp_path, f"field_provider nonesuch -> {NEW}  # 2026-10-29, #178")
    assert failures(check(root)) == [f"{root / LIST}:10: " + NOT_NEW.split(":10: ")[1]]


def test_control_a_map_line_whose_new_name_the_head_does_not_register_fails(tmp_path):
    root = renamed_since_the_base(tmp_path, f"field_provider {OLD} -> nonesuch  # 2026-10-29, #178")
    found = failures(check(root))
    assert f"{root / MAP}:1: field_provider {OLD} -> nonesuch: 'nonesuch' is not a field_provider the head registers; " \
           f"fix the map, or delete the line" in found
    assert any("is not on the list at HEAD" in line for line in found), found


def test_control_a_renamed_line_whose_reason_also_changed_fails(tmp_path):
    root = renamed_since_the_base(tmp_path, f"field_provider {OLD} -> {NEW}  # 2026-10-29, #178",
                                  head_line=f"field_provider {NEW} 0.1.0 references  # its own reason")
    assert failures(check(root)) == [
        f"{root / LIST}:10: 'field_provider {NEW} 0.1.0 references' is the base's 'field_provider {OLD} 0.1.0 "
        f"references' renamed, and its reason changed from 'present at D1b' to 'its own reason'; a renamed line keeps "
        f"its reason"]


def test_control_a_map_line_whose_old_name_the_head_still_registers_fails(tmp_path):
    root = renamed_since_the_base(tmp_path, f"field_provider {NEW} -> {NEW}  # 2026-10-29, #178")
    assert any("is still registered at the head, so it was not renamed" in line for line in failures(check(root)))
