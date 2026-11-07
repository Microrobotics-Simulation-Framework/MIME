"""The inlined Mermaid's licences: the file beside the library names, with its licence's text, every package of Mermaid's
production dependency tree that the bundle can hold (and the ones Mermaid's parser bundles into its own build), holds the
licence of every part the bundle's own comments name, and the project's notice and licence expression say what the file
holds.

**What this bounds.** That the file is the one that was written (its sha256), that each package listed below is in it with
the licence its package states, that none of the types-only packages is, and that the bundle's own licence comments (the
only ones the minifier left) are covered. It does not bound that the list is every package the bundle holds: the bundle
is minified with no comment per package, and the file's header says how the list was made (a superset, from the
dependency tree and the parser's source maps).

**Control.** The check is a function of the file's text; the file with one package's section removed, or one section's
licence changed, fails it."""

import hashlib
import re
from pathlib import Path

from mime.experiment._mermaid_page import MERMAID_FILE

ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT / "src" / "mime" / "experiment" / "_mermaid_lib"
LICENCE_SHA256 = "4961b1f3e53e0ee8dcbe11a19dfc6a336174e641b2f5439fba9f86c3e69e9660"
#: Packages the file must list, with the licence each states (khroma's package states none: its file is the MIT License).
EXPECTED = {
    "mermaid": "MIT", "d3": "ISC", "d3-sankey": "BSD-3-Clause", "d3-ease": "BSD-3-Clause", "dompurify":
    "(MPL-2.0 OR Apache-2.0)", "katex": "MIT", "cytoscape": "MIT", "dagre-d3-es": "MIT", "roughjs": "MIT", "marked": "MIT",
    "dayjs": "MIT", "stylis": "MIT", "uuid": "MIT", "lodash-es": "MIT", "khroma": "MIT", "robust-predicates": "Unlicense",
    "langium": "MIT", "chevrotain": "Apache-2.0", "chevrotain-allstar": "MIT", "vscode-jsonrpc": "MIT", "path-browserify": "MIT",
    "@mermaid-js/parser": "MIT", "@upsetjs/venn.js": "MIT", "fastdom": "MIT", "@iconify/utils": "MIT"}
SECTION = re.compile(r"^={100}\n(?P<name>\S+) (?P<version>\S+) -- (?P<licence>.+)\n={100}\n", re.MULTILINE)


def sections(text: str) -> dict[str, str]:
    """Each package's name and the licence its section states (the last version of a name that is listed twice)."""
    return {found["name"]: found["licence"] for found in SECTION.finditer(text)}


def problems(text: str) -> list[str]:
    held = sections(text)
    found = [f"{name}: not listed" for name in EXPECTED if name not in held]
    found += [f"{name}: {held[name]!r}, not {licence!r}" for name, licence in EXPECTED.items()
              if name in held and not held[name].startswith(licence)]
    found += [f"{name}: types only, and listed" for name in held if name.startswith("@types/")]
    return found


def test_the_licence_file_is_the_one_written_and_lists_each_package_with_its_licence():
    data = (LIBRARY / "LICENSE.mermaid").read_bytes()
    assert hashlib.sha256(data).hexdigest() == LICENCE_SHA256
    text = data.decode("utf-8")
    assert problems(text) == []
    assert len(sections(text)) > 70


def test_each_section_holds_licence_text_and_dompurify_holds_both_of_its_licences():
    text = (LIBRARY / "LICENSE.mermaid").read_text(encoding="utf-8")
    found = list(SECTION.finditer(text))
    assert found, "the file has sections"
    for this, following in zip(found, [*found[1:], None], strict=True):
        body = text[this.end():following.start() if following is not None else len(text)]
        assert len(body.strip()) > 100, f"{this['name']} has no licence text"
    assert "Mozilla Public License" in text and "Apache License" in text and "Version 2.0" in text


def test_the_bundles_own_licence_comments_name_only_what_the_file_holds():
    named = re.findall(r"@license ([^|*]+)\|[^*]*?(Apache|Mozilla)", (LIBRARY / MERMAID_FILE).read_text(encoding="utf-8"))
    assert [name.strip() for name, _ in named] == ["DOMPurify 3.4.12"], "a licence comment the file may not cover"
    assert sections((LIBRARY / "LICENSE.mermaid").read_text(encoding="utf-8"))["dompurify"] == "(MPL-2.0 OR Apache-2.0)"


def test_control_a_package_removed_or_a_licence_changed_is_found():
    text = (LIBRARY / "LICENSE.mermaid").read_text(encoding="utf-8")
    (start,) = [found.start() for found in SECTION.finditer(text) if found["name"] == "langium"]
    removed = text[:start] + text[start:].replace("langium 4.2.1", "langium-gone 4.2.1", 1)
    assert problems(removed) == ["langium: not listed"]
    changed = text.replace("chevrotain 11.1.2 -- Apache-2.0", "chevrotain 11.1.2 -- MIT")
    assert problems(changed) == ["chevrotain: 'MIT', not 'Apache-2.0'"]
    assert problems(text + "\n" + "=" * 100 + "\n@types/d3 7.4.3 -- MIT\n" + "=" * 100 + "\n") == ["@types/d3: types only, and listed"]


def test_notice_and_the_licence_expression_name_what_the_library_holds():
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    assert "Mermaid 11.17.2" in notice and "src/mime/experiment/_mermaid_lib/" in notice
    assert all(name in notice for name in ("D3", "KaTeX", "Chevrotain", "DOMPurify", "robust-predicates", "Unlicense"))
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'license = "Apache-2.0 AND MIT AND BSD-3-Clause AND ISC AND Unlicense"' in pyproject
    assert '"src/mime/experiment/_mermaid_lib/LICENSE.*"' in pyproject
