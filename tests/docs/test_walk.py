"""The docs walk and its ledger, each control through the real ``mime docs check`` in a fresh process over a copy of
the checkout it doctors (``checkout.py``), or with a planted entry registered there (``planted_entries.py``):

- W1, a hand-written entry file deleted, fails naming it (its passing arm: the file written and its ledger line
  deleted);
- W2, an entry of MIME's registered without its file fails naming it, so every new entry arrives with its file;
- W3, an item on the ledger that gained its source fails, naming the line to delete: an entry's file, a mechanism
  page, an asset card, and a node of MIME's given ``NodeMeta`` and a symbol table;
- W5, a registry holding no entry of MIME's fails, rather than passing having required nothing;
- W6, an entry's file naming an old version fails, naming the page and both versions; and one naming none fails;
- and the walk's (b): a page source whose id names nothing (an entry, a target kind, a node, a family or an asset
  that does not exist, or a file out of place) fails, and so does a link to such a page, while a link to an entry
  with no file of its own yet resolves.

A lab's entry without a source is counted, not failed, unless the lab opts in (the walk's ``held``), and ``toy`` is
exempt by namespace. W4's arms for the ledger are in ``test_lists.py``."""

from collections import Counter

import pytest
from checkout import ROOT, copy_checkout, docs_check, failures
from planted_nodes import ROWS, PlantedMimeNode

from mime.docs import row_key
from mime.docs._check import list_path, node_classes
from mime.docs._export import load_namespaces
from mime.docs._walk import docs_walk
from mime.registry import registered

DRAG = "entries/interaction/drag.md"


def page(version: str = "0.3.0", body: str = "What it models, and when not to use it.\n") -> str:
    return f"---\ndescribes: {version}\n---\n\n{body}"


def ledger(root):
    return root / "docs" / "unpaged.txt"


def ledger_line(root, item: str) -> int:
    lines = ledger(root).read_text(encoding="utf-8").splitlines()
    (number,) = [n for n, line in enumerate(lines, start=1) if line.split("  #")[0].strip() == item]
    return number


def unlist(root, item: str) -> None:
    lines = ledger(root).read_text(encoding="utf-8").splitlines(keepends=True)
    del lines[ledger_line(root, item) - 1]
    ledger(root).write_text("".join(lines), encoding="utf-8")


def write(root, source: str, text: str) -> None:
    path = root / "docs" / source
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def paid(root, item: str, source: str, text: str):
    """``root`` with ``source`` written and ``item``'s ledger line deleted, where the ledger lists it: the item paid
    down (the drag's was paid in phase 7g, and its file is then written over)."""
    write(root, source, text)
    if any(line.split("  #")[0].strip() == item for line in ledger(root).read_text(encoding="utf-8").splitlines()):
        unlist(root, item)
    return root


def missing(root, item: str, source: str) -> str:
    return (f"{item}: its hand-written source ({source}) is missing and {ledger(root)} does not list it; write it (the "
            f"ledger holds what merged before the walk, and a line added needs its reason logged)")


def walked_counts(entries: int = 0, nodes: int = 0, present: int = 0) -> str:
    """The walk's line for MIME, as the checkout gives it, with ``entries`` and ``nodes`` more required and
    ``present`` more present: what a planted run adds to the checkout as it stands."""
    load_namespaces()
    records = [record for record in registered() if record.namespace in ("mime", "toy", "labx")]
    required = docs_walk(ROOT / "docs", records, node_classes(records)).required
    kinds = Counter(item.item.split(" ", 1)[0] for item in required)
    return (f"of mime ({kinds['entry'] + entries} entries, {kinds['mechanism']} mechanisms, {kinds['asset']} assets, "
            f"{kinds['node'] + nodes} nodes); {sum(item.present for item in required) + present} present")


def test_w1_an_entry_file_written_and_its_ledger_line_deleted_passes(tmp_path):
    done = docs_check(paid(copy_checkout(tmp_path), "entry interaction gravity", "entries/interaction/gravity.md",
                           page("0.1.0")))
    assert done.returncode == 0, done.stdout + done.stderr
    assert walked_counts(present=1) in done.stdout


def test_w1_a_deleted_entry_file_fails_naming_it(tmp_path):
    root = paid(copy_checkout(tmp_path), "entry interaction drag", DRAG, page())
    (root / "docs" / DRAG).unlink()
    assert failures(docs_check(root)) == [missing(root, "entry interaction drag", DRAG)]


def test_w2_an_entry_of_mimes_registered_without_its_file_fails_naming_it():
    assert failures(docs_check(plant="mime_unpaged")) == [
        missing(ROOT, "entry interaction docsplant_mime_unpaged", "entries/interaction/docsplant_mime_unpaged.md")]


@pytest.mark.parametrize(("item", "source", "text"), [
    ("entry interaction gravity", "entries/interaction/gravity.md", page("0.1.0")),
    ("mechanism load", "mechanisms/load.md", "Loads sum every delivery.\n"),
    ("asset dejongh_mlp_v2", "assets/dejongh_mlp_v2.md", "A reader's note.\n"),
], ids=["an-entry", "a-mechanism", "an-asset"])
def test_w3_a_ledger_item_that_gained_its_source_fails_naming_the_line_to_delete(tmp_path, item, source, text):
    root = copy_checkout(tmp_path)
    write(root, source, text)
    assert failures(docs_check(root)) == [
        f"{ledger(root)}:{ledger_line(root, item)}: {item!r} has its source now ({source}); delete this line"]


def noded(tmp_path):
    """A copy of the checkout holding the file of a planted entry of MIME's that builds a node of MIME's (defined under
    ``mime``, outside ``mime.testing``) carrying ``NodeMeta`` and a symbol table, its rows pinned."""
    root = copy_checkout(tmp_path)
    write(root, "entries/interaction/docsplant_mime_noded.md", page("0.1.0"))
    rows = list_path(root, "symbol_rows")
    rows.write_text(rows.read_text(encoding="utf-8") + "".join(
        f"{row_key(PlantedMimeNode, row)}  # pinned at landing\n" for row in ROWS), encoding="utf-8")
    return root


def test_a_node_of_mimes_carrying_its_table_and_off_the_ledger_passes(tmp_path):
    done = docs_check(noded(tmp_path), plant="mime_noded")
    assert done.returncode == 0, done.stdout + done.stderr
    assert walked_counts(entries=1, nodes=1, present=2) in done.stdout


def test_w3_a_node_of_mimes_on_the_ledger_given_its_table_fails_naming_the_line_to_delete(tmp_path):
    root = noded(tmp_path)
    ledger(root).write_text(ledger(root).read_text(encoding="utf-8") + "node PlantedMimeNode  # planted\n",
                            encoding="utf-8")
    source = "NodeMeta and a symbol table on mime.docsplant.PlantedMimeNode"
    assert failures(docs_check(root, plant="mime_noded")) == [
        f"{ledger(root)}:{ledger_line(root, 'node PlantedMimeNode')}: 'node PlantedMimeNode' has its source now "
        f"({source}); delete this line"]


def test_a_node_of_mimes_without_its_table_and_not_on_the_ledger_fails_naming_it(tmp_path):
    root = copy_checkout(tmp_path)
    unlist(root, "node GravityNode")
    assert failures(docs_check(root)) == [missing(root, "node GravityNode", "NodeMeta and a symbol table on "
                                                                          "mime.nodes.body._gravity.GravityNode")]


EMPTIED = ("import mime.docs._check, mime.testing\nfrom mime.registry import _registry\n"
           "for key, record in list(_registry._by_key.items()):\n"
           "    if record.namespace == 'mime':\n"
           "        del _registry._by_key[key], _registry._by_class[record.cls]\n")


def test_w5_a_registry_with_no_entry_of_mimes_fails():
    found = failures(docs_check(prelude=EMPTIED))
    assert "the registry holds no entry of mime, so the walk required no entry's source and checked nothing" in found


def test_w6_an_entry_file_naming_an_old_version_fails_naming_the_page_and_both_versions(tmp_path):
    root = paid(copy_checkout(tmp_path), "entry interaction drag", DRAG, page("0.1.0"))
    path = root / "docs" / DRAG
    assert failures(docs_check(root)) == [
        f"{path}: describes drag 0.1.0, which is not registered (registered: 0.3.0); describe the registered version, "
        f"keeping the old one's history in the text",
        f"{path}: does not name drag 0.3.0, the registered version; describe it"]


def test_an_entry_file_naming_no_version_fails(tmp_path):
    root = paid(copy_checkout(tmp_path), "entry interaction drag", DRAG, "What it models.\n")
    assert failures(docs_check(root)) == [f"{root / 'docs' / DRAG}: names no version it describes; start it with "
                                          f"front matter, '---', 'describes: 0.3.0', '---'"]


@pytest.mark.parametrize(("source", "why"), [
    ("entries/interaction/no_such_law.md", "names the interaction no_such_law, which no entry registers"),
    ("entries/drag.md", "is not an entry's file, entries/<kind>/<name>.md, kind one of interaction, observation, "
                        "dynamics, field_provider, controller, drive, geometry"),
    ("mechanisms/friction.md", "names the target kind friction, which is none of load, kinematic, operator, "
                               "equality_constraint, inequality_constraint, registration"),
    ("nodes/NoSuchNode.md", "names the node NoSuchNode, which no entry builds"),
    ("families/no_such_family.md", "names the family no_such_family, which is not registered"),
    ("assets/no_such_asset.md", "names the asset no_such_asset, which MIME does not package"),
    ("assets/notes.txt", "is not a page source (a page source is a .md file)"),
], ids=["a-missing-entry", "an-entry-out-of-place", "a-missing-target-kind", "a-missing-node", "a-missing-family",
        "a-missing-asset", "not-markdown"])
def test_a_page_source_whose_id_names_nothing_fails(tmp_path, source, why):
    root = copy_checkout(tmp_path)
    write(root, source, "A note.\n")
    assert failures(docs_check(root)) == [f"{root / 'docs' / source}: {why}"]


def test_a_link_to_an_entry_without_its_own_file_yet_resolves(tmp_path):
    done = docs_check(paid(copy_checkout(tmp_path), "entry interaction drag", DRAG, page(
        body="Beside [gravity](gravity.md#settings) and [the load](../../mechanisms/load.md).\n")))
    assert done.returncode == 0, done.stdout + done.stderr


def test_a_link_to_a_page_that_names_nothing_fails(tmp_path):
    root = paid(copy_checkout(tmp_path), "entry interaction drag", DRAG,
                page(body="Beside [lift](lift.md) and [the guide](../../guide/x.md).\n"))
    assert failures(docs_check(root)) == [
        f"{root / 'docs' / DRAG}: its link 'lift.md' names the interaction lift, which no entry registers"]


def test_a_labs_entry_without_a_source_is_counted_and_fails_once_the_lab_opts_in(tmp_path):
    """The walk itself, over a copy of the docs root: by default the lab's entries, ``labx.gantry`` and ``labx.nudge``,
    are counted with their nodes; held to the walk, their entries and nodes are required, and fail; ``toy`` is exempt
    by namespace throughout."""
    docs = copy_checkout(tmp_path) / "docs"
    load_namespaces()
    records = [record for record in registered() if record.namespace in ("mime", "toy", "labx")]
    walked = docs_walk(docs, records, node_classes(records))
    assert walked.problems == []
    assert "outside MIME, without a hand-written source here, counted, not failed: labx 2 (dynamics labx.gantry " \
           "0.1.0, interaction labx.nudge 0.1.0); nodes without NodeMeta: 2 (GantryNode, NudgeNode)" in walked.summary
    assert "exempt from the hand-written half by namespace (toy): 12 entries" in walked.summary
    held = docs_walk(docs, records, node_classes(records), held=("mime", "labx"))
    assert [problem for problem in held.problems if "labx" in problem or "Nudge" in problem or "Gantry" in problem] == [
        missing(docs.parent, "entry dynamics labx.gantry", "entries/dynamics/labx.gantry.md"),
        missing(docs.parent, "entry interaction labx.nudge", "entries/interaction/labx.nudge.md"),
        missing(docs.parent, "node GantryNode", "NodeMeta and a symbol table on mime_lab_fixture.GantryNode"),
        missing(docs.parent, "node NudgeNode", "NodeMeta and a symbol table on mime_lab_fixture.NudgeNode")]


# -- each requirement and the ledger's own arms ----------------------------------------------------------------------

def test_a_ledger_line_naming_nothing_the_walk_requires_fails_naming_it(tmp_path):
    root = copy_checkout(tmp_path)
    ledger(root).write_text(ledger(root).read_text(encoding="utf-8") + "entry interaction no_such_law  # planted\n",
                            encoding="utf-8")
    assert failures(docs_check(root)) == [
        f"{ledger(root)}:{ledger_line(root, 'entry interaction no_such_law')}: 'entry interaction no_such_law' is "
        f"nothing the walk requires a source for; delete this line"]


@pytest.mark.parametrize(("item", "source"), [
    ("mechanism load", "mechanisms/load.md"),
    ("asset dejongh_mlp_v2", "assets/dejongh_mlp_v2.md"),
], ids=["a-mechanism", "an-asset"])
def test_a_mechanism_or_an_asset_without_its_page_and_off_the_ledger_fails_naming_it(tmp_path, item, source):
    root = copy_checkout(tmp_path)
    unlist(root, item)
    assert failures(docs_check(root)) == [missing(root, item, source)]


def test_a_malformed_ledger_line_fails_naming_the_ledgers_form(tmp_path):
    root = copy_checkout(tmp_path)
    ledger(root).write_text(ledger(root).read_text(encoding="utf-8") + "page drag  # planted\n", encoding="utf-8")
    number = ledger_line(root, "page drag")
    assert failures(docs_check(root)) == [
        f"{ledger(root)}:{number}: 'page drag' is malformed: an item is 'entry <kind> <name>', 'mechanism <target "
        f"kind>', 'node <class>' or 'asset <name>'",
        f"{ledger(root)}:{number}: 'page drag' is nothing the walk requires a source for; delete this line"]


def test_a_node_of_mimes_with_nodemeta_and_no_symbol_table_is_not_documented(tmp_path):
    """The walk's (a) asks for both: ``NodeMeta`` alone, its equations needing no row, leaves the node's source
    missing."""
    root = copy_checkout(tmp_path)
    write(root, "entries/interaction/docsplant_mime_meta_only.md", page("0.1.0"))
    assert failures(docs_check(root, plant="mime_meta_only")) == [
        missing(root, "node PlantedMimeMetaOnly", "NodeMeta and a symbol table on mime.docsplant.PlantedMimeMetaOnly")]


# -- the link forms and the front matter ------------------------------------------------------------------------------

@pytest.mark.parametrize("link", [
    "[lift][l].\n\n[l]: lift.md",
    '<a href="lift.md">lift</a>.',
    "[lift](<lift.md>).",
    '[lift](lift.md "the lift").',
], ids=["a-reference-definition", "an-html-href", "in-angle-brackets", "with-a-title"])
def test_each_link_form_to_a_page_that_names_nothing_fails(tmp_path, link):
    root = paid(copy_checkout(tmp_path), "entry interaction drag", DRAG, page(body=f"Beside {link}\n"))
    assert failures(docs_check(root)) == [
        f"{root / 'docs' / DRAG}: its link 'lift.md' names the interaction lift, which no entry registers"]


@pytest.mark.parametrize(("front", "why"), [
    ("describes: [0.2.0", ["its front matter is not YAML (the file's line 2): expected ',' or ']', but got '<stream "
                           "end>'; fix it"]),
    ("describes: 0.2", ["describes 0.2, which YAML reads as a float, not a version; quote it, as 'describes: "
                        "\"0.3.0\"'"]),
], ids=["not-yaml", "a-version-read-as-a-number"])
def test_a_front_matter_not_yaml_or_a_version_read_as_a_number_fails_naming_the_file(tmp_path, front, why):
    """Named, not a traceback. A two-part version is a number token too, which the number rule names."""
    root = paid(copy_checkout(tmp_path), "entry interaction drag", DRAG, f"---\n{front}\n---\n\nWhat it models.\n")
    found = failures(docs_check(root))
    assert found[:1] == [f"{root / 'docs' / DRAG}: {why[0]}"]
    assert found[1:] == ([] if front.endswith("0.2.0") else [
        f"docs/{DRAG}:2: the number token(s) 0.2 in prose. A number is a claim record's (see MIME-ANO-nnn); or list "
        f"the line as 'docs/{DRAG}  0.2  #1  # <reason>' in docs/ratchets/number_allowlist.txt"])
