"""The executable-docs check (``mime docs check``'s stage over ``docs/``): what a page shows is included from a file a
test names, an output a script printed, or a fragment with its reason; a ``mime`` command line is one the CLI's own
parser accepts; a link resolves, its anchor too; the pages that read a dataclass or the parser are held equal to them.

Each planted break below is a tree of its own with one defect, and fails every run: the checks are deterministic, and no
arm draws anything. Each has its control, the same tree without the defect, which must pass, so a plant fails for its own
reason and not because the tree was bad. The scripts of the output blocks are stood in for by a function that returns
text, so these arms run no JAX; ``test_tutorials.py`` runs the real tutorials, and the wiring arms at the end run the real
``mime docs check`` over a doctored copy of the checkout.

**What this bounds.** That each named defect is found, and that a clean tree is not flagged for it. It does not bound what
the check cannot see: a block marked a fragment with a reason that is false, or prose that says a wrong thing about a
command that parses."""

import dataclasses
import re
import shutil
from pathlib import Path

import pytest
from checkout import ROOT, copy_checkout, docs_check, edit, failures

from mime.build import BuildReport
from mime.docs._executable import (
    GLOSSARY_URL,
    anchors,
    executable_report,
    expand,
    hand_written,
    link_problems,
    parse,
    refresh_outputs,
    slug,
)
from mime.docs._generated import (
    BUILD_REPORT_PAGE,
    GENERATORS,
    ONLY_GENERATED,
    build_report_problems,
    cli_reference,
    only_generated_problems,
)

GENERATED = frozenset(GENERATORS)
PRINTED = "first line\nsecond line\n"


def tree(tmp_path: Path, page: str, *, name: str = "docs/guide/page.md", extra: dict[str, str] | None = None) -> Path:
    """A small checkout with one page, a script it can name, a test that names the script, and the lab fixture."""
    files = {name: page, "examples/a.py": "print('x')\n", "tests/test_names.py": "# examples/a.py\n"}
    files |= extra or {}
    for relative, text in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    shutil.copytree(ROOT / "lab_fixture", tmp_path / "lab_fixture", ignore=shutil.ignore_patterns("__pycache__"))
    return tmp_path


def problems(root: Path, pages: dict[str, str] | None = None) -> list[str]:
    return executable_report(root, pages or {}, runner=lambda path: PRINTED, generated=GENERATED).problems


# -- a fenced block ---------------------------------------------------------------------------------------------------

def test_a_marked_fragment_with_a_reason_passes(tmp_path):
    root = tree(tmp_path, "# P\n\n<!-- mime-fragment: a line of the reader's own script -->\n```python\nx = 1\n```\n")
    assert problems(root) == []


def test_an_unmarked_fenced_block_fails_naming_its_line(tmp_path):
    root = tree(tmp_path, "# P\n\ntext\n\n```python\nx = 1\n```\n")
    (found,) = problems(root)
    assert found.startswith("docs/guide/page.md:5: a fenced block is neither included") and "mime-fragment" in found


def test_a_fragment_with_no_reason_fails(tmp_path):
    root = tree(tmp_path, "# P\n\n<!-- mime-fragment:  -->\n```python\nx = 1\n```\n")
    (found,) = problems(root)
    assert found.startswith("docs/guide/page.md:4: a fragment is marked with no reason")


def test_a_marker_that_no_block_follows_fails(tmp_path):
    root = tree(tmp_path, "# P\n\n<!-- mime-fragment: a reason -->\n\nprose between\n")
    (found,) = problems(root)
    assert "is followed by no fenced block" in found


def test_a_fence_never_closed_fails(tmp_path):
    root = tree(tmp_path, "# P\n\n<!-- mime-fragment: a reason -->\n```python\nx = 1\n")
    assert any("is never closed" in found for found in problems(root))


# -- a command --------------------------------------------------------------------------------------------------------

def test_a_command_of_the_parser_passes_in_a_block_and_in_a_span(tmp_path):
    page = ("# P\n\nRun `mime run my_experiment --duration 10s --out out`, or\n\n<!-- mime-fragment: shell lines -->\n"
            "```sh\nuv run --no-sync mime docs check --base origin/release/v0.3.0 \\\n  --skip-hash-seed-builds\n"
            "mime env check x --variant cpu | tee log\n```\n")
    assert problems(tree(tmp_path, page)) == []


def test_a_documented_flag_the_parser_does_not_have_fails_naming_the_line_and_the_command(tmp_path):
    page = "# P\n\n<!-- mime-fragment: shell -->\n```sh\nmime docs check --no-such-flag\n```\n"
    (found,) = problems(tree(tmp_path, page))
    assert found.startswith("docs/guide/page.md:4: `mime docs check --no-such-flag`: --no-such-flag is not a flag of "
                            "`mime docs check`")


def test_a_documented_command_the_parser_does_not_have_fails(tmp_path):
    (found,) = problems(tree(tmp_path, "# P\n\nThen `mime docs lint` prints it.\n"))
    assert "`mime docs` has no command lint" in found


# -- an output block --------------------------------------------------------------------------------------------------

OUTPUT = "# P\n\n<!-- mime-output: examples/a.py -->\n```text\n{}```\n"


def test_an_output_block_that_is_what_the_script_printed_passes(tmp_path):
    assert problems(tree(tmp_path, OUTPUT.format(PRINTED))) == []


def test_an_output_block_edited_by_one_character_fails_naming_the_first_difference(tmp_path):
    (found,) = problems(tree(tmp_path, OUTPUT.format(PRINTED.replace("second", "secund"))))
    assert "docs/guide/page.md:4: the output block is not what examples/a.py prints" in found
    assert "'second line'" in found and "'secund line'" in found


def test_an_output_block_naming_a_script_that_does_not_exist_fails(tmp_path):
    root = tree(tmp_path, OUTPUT.format(PRINTED).replace("examples/a.py", "examples/gone.py"))
    (found,) = problems(root)
    assert "names examples/gone.py, which is no file" in found


def test_a_script_that_fails_is_a_problem_naming_it(tmp_path):
    root = tree(tmp_path, OUTPUT.format(PRINTED))

    def broken(path: Path) -> str:
        raise RuntimeError("boom")

    found = executable_report(root, {}, runner=broken, generated=GENERATED).problems
    assert any("examples/a.py: could not be run to regenerate its output block: RuntimeError: boom" in f for f in found)


def test_a_stale_output_block_is_rewritten_by_the_refresh(tmp_path):
    root = tree(tmp_path, OUTPUT.format("old\n"))
    assert refresh_outputs(root, runner=lambda path: PRINTED) == ["docs/guide/page.md"]
    assert problems(root) == []


def test_a_command_in_a_code_span_wrapped_over_two_lines_is_read_and_a_stale_one_fails_at_its_first_line(tmp_path):
    page = "# P\n\nRun `mime docs\ncheck --base origin/release/v0.3.0` or `mime docs\ncheck --no-such-flag` to see.\n"
    (found,) = problems(tree(tmp_path, page))
    assert found.startswith("docs/guide/page.md:4: `mime docs check --no-such-flag`: --no-such-flag is not a flag of")


def test_a_span_that_is_a_message_and_not_a_command_is_not_read_as_one(tmp_path):
    assert problems(tree(tmp_path, "# P\n\nIt says `mime serve: no client has\nbeen connected; stopping` and exits.\n")) == []


@pytest.mark.parametrize("form", ["> ```sh\n> mime docs check\n> ```", "<pre>\nx = 1\n</pre>", "A <code>x = 1</code> here."])
def test_code_in_a_form_the_check_does_not_read_fails(tmp_path, form):
    (found, *_) = problems(tree(tmp_path, f"# P\n\n{form}\n"))
    assert "is not read by this check" in found


def test_control_those_forms_named_in_a_code_span_are_prose(tmp_path):
    assert problems(tree(tmp_path, "# P\n\nA `<pre>` element and a `> ```sh` quotation are not read.\n")) == []


def test_an_output_block_may_name_a_script_of_examples_only(tmp_path):
    root = tree(tmp_path, OUTPUT.format(PRINTED).replace("examples/a.py", "tests/test_names.py"))
    (found,) = problems(root)
    assert "the output block names tests/test_names.py; a script whose output a guide shows is one of examples/" in found
    assert problems(tree(tmp_path / "control", OUTPUT.format(PRINTED))) == []


# -- an include -------------------------------------------------------------------------------------------------------

def test_an_include_of_a_file_a_test_names_is_replaced_by_its_text_in_a_fenced_block(tmp_path):
    root = tree(tmp_path, "# P\n\n<!-- mime-include: examples/a.py -->\n")
    assert problems(root) == []
    text, found = expand((root / "docs/guide/page.md").read_text(encoding="utf-8"), root, lambda name: "")
    assert found == [] and "```python\nprint('x')\n```" in text and "mime-include" not in text


def test_an_include_of_a_file_no_test_names_fails(tmp_path):
    root = tree(tmp_path, "# P\n\n<!-- mime-include: examples/b.py -->\n", extra={"examples/b.py": "y = 2\n"})
    (found,) = problems(root)
    assert "no test names examples/b.py" in found


def test_an_include_of_a_region_or_a_symbol_that_is_not_there_fails(tmp_path):
    root = tree(tmp_path, "# P\n\n<!-- mime-include: examples/a.py#nowhere -->\n<!-- mime-include: examples/a.py::Gone -->\n")
    found = problems(root)
    assert any("has no '# region: nowhere'" in f for f in found) and any("has no class or function Gone" in f
                                                                       for f in found)


def test_a_region_and_a_symbol_are_cut_out_with_their_markers_left_behind(tmp_path):
    source = "# region: one\nvalue = 1\n# endregion one\n\n@decorated\nclass Thing:\n    x = 1\n\n\nz = 3\n"
    root = tree(tmp_path, "# P\n\n<!-- mime-include: examples/a.py#one -->\n<!-- mime-include: examples/a.py::Thing -->\n",
                extra={"examples/a.py": source})
    text, found = expand((root / "docs/guide/page.md").read_text(encoding="utf-8"), root, lambda name: "")
    assert found == [] and "```python\nvalue = 1\n```" in text and "```python\n@decorated\nclass Thing:\n    x = 1\n```" in text
    assert "region" not in text.replace("*From", "")


LAB = "docs/guide/lab_author.md"


def test_a_lab_author_example_from_the_ci_built_fixture_passes(tmp_path):
    page = "# L\n\n<!-- mime-include: lab_fixture/src/mime_lab_fixture/__init__.py::Nudge -->\n"
    assert problems(tree(tmp_path, page, name=LAB)) == []


def test_a_lab_author_example_that_is_not_the_fixtures_file_fails(tmp_path):
    page = "# L\n\n<!-- mime-include: examples/a.py -->\n"
    (found,) = problems(tree(tmp_path, page, name=LAB))
    assert "includes only from lab_fixture/, the package CI builds and installs; examples/a.py is not there" in found


def test_a_lab_author_include_that_leaves_the_fixture_by_dotdot_fails(tmp_path):
    """The path is resolved before it is compared: ``lab_fixture/../examples/a.py`` starts with ``lab_fixture/`` and is
    not in it. The control above is the fixture's own file."""
    page = "# L\n\n<!-- mime-include: lab_fixture/../examples/a.py -->\n"
    (found,) = problems(tree(tmp_path, page, name=LAB))
    assert "includes only from lab_fixture/, the package CI builds and installs; examples/a.py is not there" in found


def test_the_fixture_must_still_be_built_by_ci(tmp_path):
    page = "# L\n\n<!-- mime-include: lab_fixture/pyproject.toml -->\n"
    root = tree(tmp_path, page, name=LAB, extra={".github/workflows/ci.yml": "steps:\n  - run: uv sync\n"})
    (found,) = problems(root)
    assert "CI no longer builds and installs ./lab_fixture" in found


# -- a generated text -------------------------------------------------------------------------------------------------

def test_a_generate_directive_naming_no_generator_fails(tmp_path):
    (found,) = problems(tree(tmp_path, "# P\n\n<!-- mime-generate: nothing-of-the-kind -->\n"))
    assert "no generated text is named 'nothing-of-the-kind'" in found


# -- a link -----------------------------------------------------------------------------------------------------------

def test_a_link_that_resolves_passes_in_a_source_an_anchor_and_a_generated_page(tmp_path):
    page = ("# P\n\n## Part two\n\nSee [the other](other.md#the-heading), [here](#part-two), [the vocabulary]"
            "(../vocabulary.md#names), [a folder](../validation/) and [outside](https://example.org/x#y).\n")
    root = tree(tmp_path, page, extra={"docs/guide/other.md": "# O\n\n## The heading\n", "docs/validation/x.md": "# V\n"})
    assert link_problems(root, hand_written(root), {"pages/vocabulary.md": "# Vocabulary\n\n## Names\n"}) == []


def test_a_broken_relative_link_fails_naming_the_line(tmp_path):
    root = tree(tmp_path, "# P\n\nSee [gone](missing.md).\n")
    (found,) = link_problems(root, hand_written(root), {})
    assert found.startswith("docs/guide/page.md:3: the link 'missing.md' names no source, and no page the build generates")


def test_a_broken_anchor_fails_in_a_source_in_this_file_and_in_a_generated_page(tmp_path):
    page = "# P\n\n[a](other.md#nowhere) [b](#nowhere) [c](../vocabulary.md#nowhere)\n"
    root = tree(tmp_path, page, extra={"docs/guide/other.md": "# O\n"})
    found = link_problems(root, hand_written(root), {"pages/vocabulary.md": "# Vocabulary\n"})
    assert [f.split(" names ")[0].split(": ")[1] for f in found] == ["the link 'other.md#nowhere'",
                                                                      "the link '#nowhere'",
                                                                      "the link '../vocabulary.md#nowhere'"]


@pytest.mark.parametrize("url", ["https://microrobotica.org/glossary/", "https://microrobotica.org/mime/glossary.html",
                                 "http://microrobotica.org/glossary.html", "https://example.org/Glossary"])
def test_a_glossary_link_that_is_not_the_one_constant_fails_naming_the_line(tmp_path, url):
    root = tree(tmp_path, f"# P\n\nThe [terms]({url}) are shared.\n")
    (found,) = link_problems(root, hand_written(root), {})
    assert found.startswith(f"docs/guide/page.md:3: the link '{url}' is a glossary") and GLOSSARY_URL in found


def test_the_glossary_link_of_the_constant_passes_and_so_does_another_external_link(tmp_path):
    root = tree(tmp_path, f"# P\n\nThe [terms]({GLOSSARY_URL}) and [other](https://example.org/x) are shared.\n")
    assert link_problems(root, hand_written(root), {}) == []


def test_every_glossary_link_in_the_docs_is_the_constant_and_the_front_page_has_one():
    """Over the real files, every Markdown file under ``docs/`` and the READMEs: the constant is the canonical page (not
    the ``/glossary/`` that answers 404), and the guide's front page links it."""
    assert GLOSSARY_URL == "https://microrobotica.org/glossary.html"
    pattern = re.compile(r"https?://[^\s)>\]]*glossary[^\s)>\]]*", re.IGNORECASE)
    found = {path: pattern.findall(path.read_text(encoding="utf-8"))
             for path in [*(ROOT / "docs").rglob("*.md"), *ROOT.glob("*.md")]}
    assert all(link == GLOSSARY_URL for links in found.values() for link in links), found
    assert GLOSSARY_URL in (ROOT / "docs" / "guide" / "index.md").read_text(encoding="utf-8")


def test_a_pandoc_citation_before_a_colon_is_no_link_definition(tmp_path):
    root = tree(tmp_path, "# P\n\n[@Smith2018; @Gallagher]: the kind samples its surface.\n")
    assert link_problems(root, hand_written(root), {}) == []


def test_anchors_follow_github_slugs():
    assert slug("Precision is per `rig`") == "precision-is-per-rig"
    assert anchors("# A\n\n## Same\n\n## Same\n\n```\n## inside a fence\n```\n") == {"a", "same", "same-1"}


# -- the pages that read the code -------------------------------------------------------------------------------------

def report_page(tmp_path: Path, headings: list[str]) -> Path:
    page = tmp_path / BUILD_REPORT_PAGE
    page.parent.mkdir(parents=True)
    page.write_text("# R\n\n" + "".join(f"## {heading}\n\ntext\n\n" for heading in headings), encoding="utf-8")
    return tmp_path


FIELDS = [f"`{f.name}`" for f in dataclasses.fields(BuildReport)]


def test_a_build_report_page_with_one_heading_per_field_passes(tmp_path):
    assert build_report_problems(report_page(tmp_path, FIELDS)) == []


def test_a_heading_that_is_not_a_field_fails_and_so_does_a_field_with_no_heading(tmp_path):
    found = build_report_problems(report_page(tmp_path, [*FIELDS[1:], "`invented`", "Plain words"]))
    assert len(found) == 3
    assert any("the heading `invented` is not a field of BuildReport" in f for f in found)
    assert any("'Plain words' is not a field name" in f for f in found)
    assert any(f"BuildReport's field `{FIELDS[0].strip('`')}` has no heading" in f for f in found)


def test_a_field_with_two_headings_fails(tmp_path):
    (found,) = build_report_problems(report_page(tmp_path, [*FIELDS, FIELDS[0]]))
    assert "has 2 headings" in found


def test_the_generated_reference_page_is_one_directive_and_a_hand_edit_to_it_fails(tmp_path):
    (relative,) = ONLY_GENERATED
    page = tmp_path / relative
    page.parent.mkdir(parents=True)
    page.write_text((ROOT / relative).read_text(encoding="utf-8"), encoding="utf-8")
    assert only_generated_problems(tmp_path) == []
    page.write_text(page.read_text(encoding="utf-8") + "\nA sentence a hand added.\n", encoding="utf-8")
    (found,) = only_generated_problems(tmp_path)
    assert found.startswith(f"{relative}: it is the generated cli-reference and nothing else") and "A sentence" in found


def test_the_real_pages_that_read_the_code_pass_their_checks():
    assert build_report_problems(ROOT) == [] and only_generated_problems(ROOT) == []


# -- the generated reference ------------------------------------------------------------------------------------------

def test_the_cli_reference_names_every_command_and_flag_the_parser_has():
    from mime.cli import build_parser
    from mime.cli._completion import describe, _walk  # pyright: ignore[reportPrivateUsage]

    text = cli_reference(ROOT)
    tree_ = describe(build_parser())
    for node in _walk(tree_):
        if node.words:
            assert f"`mime {' '.join(node.words)}`" in text, node.words
        for option in node.options:
            if option.strings != ("-h", "--help"):
                assert all(f"`{string}`" in text for string in option.strings), (node.words, option.strings)
    assert text == cli_reference(ROOT), "the same parser gives the same text"


def test_the_profiles_text_names_every_profile_and_the_rules_the_tables_hold():
    from mime.validity import PROFILE_RULES, Profile

    text = GENERATORS["profiles"](ROOT)
    assert all(f"`{profile.value}`" in text for profile in Profile)
    assert all(f"`{name}`" in text for _, name in PROFILE_RULES.rules)


def test_the_limits_text_lists_the_entries_that_declare_limits_and_no_other():
    text = GENERATORS["limits"](ROOT)
    assert "**interaction `contact`" in text and "**interaction `gravity`" not in text


# -- the real command, over a doctored copy of the checkout -----------------------------------------------------------

def doctored(tmp_path: Path) -> Path:
    return copy_checkout(tmp_path)


def test_the_command_fails_an_unmarked_block_naming_its_file_and_line(tmp_path):
    root = doctored(tmp_path)
    page = root / "docs" / "guide" / "install.md"
    page.write_text(page.read_text(encoding="utf-8") + "\n```python\nx = 1\n```\n", encoding="utf-8")
    problem = [f for f in failures(docs_check(root)) if "install.md" in f and "fenced block" in f]
    assert len(problem) == 1 and problem[0].startswith("docs/guide/install.md:")


def test_the_command_fails_a_cli_reference_edited_by_hand(tmp_path):
    root = doctored(tmp_path)
    page = root / "docs" / "guide" / "cli.md"
    page.write_text(page.read_text(encoding="utf-8") + "\nmime run does everything.\n", encoding="utf-8")
    assert any(f.startswith("docs/guide/cli.md: it is the generated cli-reference") for f in failures(docs_check(root)))


def test_the_command_fails_a_stale_flag_in_a_guide(tmp_path):
    root = doctored(tmp_path)
    edit(root / "docs" / "guide" / "install.md", "mime env check my_experiment --variant cpu",
         "mime env check my_experiment --variant cpu --nonexistent")
    assert any("--nonexistent is not a flag of `mime env check`" in f for f in failures(docs_check(root)))


def test_the_command_fails_a_broken_link_and_a_broken_anchor_in_a_guide(tmp_path):
    root = doctored(tmp_path)
    edit(root / "docs" / "guide" / "index.md", "[Tutorial A](tutorial_a.md)", "[Tutorial A](tutorial_gone.md)")
    edit(root / "docs" / "guide" / "index.md", "(install.md)", "(install.md#no-such-heading)")
    found = failures(docs_check(root))
    assert any("the link 'tutorial_gone.md' names no source" in f for f in found)
    assert any("the link 'install.md#no-such-heading' names no heading of install.md" in f for f in found)


def test_the_command_passes_the_guides_of_the_checkout_but_for_what_these_tests_doctored():
    done = docs_check(ROOT)
    assert [f for f in done.stdout.splitlines() if f.startswith("FAIL") and "docs/guide/" in f] == []
    assert any(line.startswith("executable docs: ") and "scripts not run" in line for line in done.stdout.splitlines())


def test_an_anchor_lands_on_the_built_page_when_the_source_is_printed_inside_one(tmp_path):
    root = tree(tmp_path, "# P\n\nSee [the entry](../entries/thing/x.md#settings).\n",
                extra={"docs/entries/thing/x.md": "# What it models\n"})
    assert link_problems(root, hand_written(root), {}) != []
    built = {"pages/entries/thing/x.md": "# X\n\n## Settings\n\n### What it models\n"}
    assert link_problems(root, hand_written(root), built) == []
