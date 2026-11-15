"""The files of ``mime scaffold plugin``: the layout of a typical MIME plugin, every file short and commented as optional.

The layout is the clot plugin's (the reference plugin), kept to what a plugin needs: the entry points (``mime.entries``
and ``mime.assets``), an adapter package (``entries/``) over an engine (``engine/``) and a second package (``fit/``) that
import no ``mime``, a records folder with its data, tests (the registry's round trip, node verification and parameter reach
on the stub entry, validity lines, the two import rules (no ``mime`` in the engine and ``fit``; the package imports itself
relatively) each with its control, a Hypothesis property with one planted error, a benchmark in a slow lane, an
``assert_type`` file and the duration guard's own test), a benchmark folder with ``mime.toml``, an example, the README's
"Use" section, the citation and licence, a typing configuration (strict, with the unknown-type rules off for the packages
that import no ``mime``) and CI, with a duration guard in its default lane. The namespace is one value, filled in the six places it appears; no file holds
a machine path.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from importlib.metadata import version as installed_version
from typing import NamedTuple, cast

from packaging.version import Version

from mime.push._defaults import Defaults
from mime.push._manifest import BUNDLE, manifest_text
from mime.push._sharing import DEFAULT_LICENCE
from mime.push._toml import value_text
from mime.scaffold._templates import fill, template
from mime.scaffold._values import (
    AUTHORS,
    DISTRIBUTION,
    NAMESPACE,
    PACKAGE,
    PLUGIN_LICENCE,
    Held,
    mime_range,
    person_name,
)

#: The licence whose full text ships with the template; any other gets a file that names it and says to put its text there.
SHIPPED_LICENCE = "Apache-2.0"
#: A creator where nothing names one: the record's own default, to replace.
_PLACEHOLDER = {"family": "Author", "given": "A."}


class PluginFile(NamedTuple):
    """One file of the plugin: its path below the folder (``${package}`` filled), the template that writes it, and the note
    its line in the tree carries."""

    path: str
    template: str
    note: str


#: Each file of the plugin: its path below the folder (``${package}`` filled), the template that writes it, and the note
#: its line in the tree carries. A file no template writes (``benchmarks/mime.toml``, ``LICENSE``, the records) is made
#: in :func:`plugin_files`.
LAYOUT: tuple[PluginFile, ...] = (
    PluginFile("pyproject.toml", "pyproject.toml.txt", "licence, the mime-engine dependency, the entry points, the pytest lanes"),
    PluginFile("README.md", "README.md.txt", "Use (CI runs its `bash ci` blocks), the naming rule, the layout, the tests"),
    PluginFile(".gitignore", "gitignore.txt", "optional"),
    PluginFile("CITATION.cff", "CITATION.cff.txt", "how to cite it; creators from your defaults"),
    PluginFile("pyrightconfig.json", "pyrightconfig.json.txt",
               "pyright strict, the unknown-type rules off for the engine and fit alone, and the environment it reads"),
    PluginFile(".github/workflows/ci.yml", "ci.yml.txt", "default, slow (label-gated), weekly against MIME's release branch, typing"),
    PluginFile("src/${package}/__init__.py", "init.py.txt", ""),
    PluginFile("src/${package}/engine/__init__.py", "engine_init.py.txt", "your law as pure JAX: imports no `mime`"),
    PluginFile("src/${package}/engine/load.py", "engine_load.py.txt", "the example law: replace it"),
    PluginFile("src/${package}/fit/__init__.py", "fit_init.py.txt", "a second package that imports no `mime`, empty: a fitting harness"),
    PluginFile("src/${package}/entries/__init__.py", "entries_init.py.txt", "the `mime.entries` entry point's module"),
    PluginFile("src/${package}/entries/_vocabulary.py", "entries_vocabulary.py.txt", "the namespace, and the naming rule"),
    PluginFile("src/${package}/entries/example_load.py", "entries_example_load.py.txt", "one example entry: replace it"),
    PluginFile("src/${package}/records/__init__.py", "records_init.py.txt", "the `mime.assets` entry point's module, and the tiers"),
    PluginFile("src/${package}/records/data/example.json", "records_example.json.txt", "the bytes the example record names"),
    PluginFile("tests/conftest.py", "conftest.py.txt", "one core and the CPU; the planted error from the environment"),
    PluginFile("tests/test_registry.py", "test_registry.py.txt", "the registry walk, the saved round trip, the record's hash"),
    PluginFile("tests/test_nodes.py", "test_nodes.py.txt", "node verification and parameter reach on the entry's node"),
    PluginFile("tests/test_validity.py", "test_validity.py.txt", "every entry has validity lines, none empty"),
    PluginFile("tests/import_rules.py", "import_rules.py.txt", "what the two import tests read from a source file"),
    PluginFile("tests/test_engine_imports.py", "test_engine_imports.py.txt",
               "the engine and fit import no `mime`, with a planted import"),
    PluginFile("tests/test_relative_imports.py", "test_relative_imports.py.txt",
               "the package imports itself relatively, with a planted absolute import"),
    PluginFile("tests/test_durations.py", "test_durations.py.txt", "the duration guard fails a slow test, and passes a listed one"),
    PluginFile("tests/duration_allowlist.txt", "duration_allowlist.txt", "the tests that may be slow, each with its reason"),
    PluginFile("scripts/check_durations.py", "check_durations.py.txt", "CI's duration guard: a test over 30 s must be listed"),
    PluginFile("tests/plants.py", "plants.py.txt", "planted errors, for controls"),
    PluginFile("tests/engine/test_properties.py", "engine_test_properties.py.txt", "Hypothesis properties, and the control"),
    PluginFile("tests/verification/test_example.py", "verification_test_example.py.txt", "a `@verification_benchmark`, slow lane"),
    PluginFile("tests/assert_types.py", "assert_types.py.txt", "types an editor completes from, held by pyright"),
    PluginFile("benchmarks/build_case.py", "benchmarks_build_case.py.txt", "saves the example with its mime.toml"),
    PluginFile("examples/example.py", "examples_example.py.txt", "what a user runs"),
)


def release_branch() -> str:
    """MIME's release branch for the installed version (``0.3.0.dev1``: ``release/v0.3.0``), which the weekly lane of a
    plugin's CI is run against."""
    release = Version(installed_version("mime-engine")).release
    return f"release/v{release[0]}.{release[1]}.0"


def _yaml(value: str) -> str:
    """A text as a YAML double-quoted scalar (JSON's escapes are YAML's)."""
    return json.dumps(value, ensure_ascii=False)


def _cff_authors(people: list[dict[str, str]]) -> str:
    lines = ["authors:"]
    for person in people or [_PLACEHOLDER]:
        lines.append(f"  - family-names: {_yaml(person['family'])}")
        if "given" in person:
            lines.append(f"    given-names: {_yaml(person['given'])}")
        if "orcid" in person:
            lines.append(f"    orcid: {_yaml('https://orcid.org/' + person['orcid'])}")
    if not people:
        lines.append("# a placeholder: replace it with the people who wrote the code")
    return "\n".join(lines)


def plugin_files(held: Mapping[str, Held], defaults: Defaults) -> tuple[dict[str, str], dict[str, str]]:
    """Every file of the plugin by its path below the folder with its text, and each path's note for the tree.

    Every value is filled from ``held`` (``--set``, the person's defaults or scaffold's own) and ``defaults`` (the
    content licence and creators the person keeps); a template names the values it takes, and one it is not given is an
    error. A value is written as the file's own language reads it: TOML, YAML, JSON or Python."""
    people = cast(list[dict[str, str]], held[AUTHORS].value) if AUTHORS in held else []
    names = [person_name(person) for person in people]
    distribution, namespace, package = (str(held[key].value) for key in (DISTRIBUTION, NAMESPACE, PACKAGE))
    licence = str(held[PLUGIN_LICENCE].value)
    content = defaults.licences.get("content", DEFAULT_LICENCE)
    data = fill("plugin", "records_example.json.txt", {})
    values = {
        "distribution": distribution, "namespace": namespace, "package": package, "cff_licence": _yaml(licence),
        "prefix": namespace.upper().replace("_", "-"), "release_branch": release_branch(), "cff_title": _yaml(distribution),
        "content_licence": json.dumps(content), "authors": _cff_authors(people),
        "sha256": hashlib.sha256(data.encode("utf-8")).hexdigest(), "size": str(len(data.encode("utf-8"))),
        "credit": json.dumps(", ".join(names) or "the plugin's authors", ensure_ascii=False)}
    values["record_licence"] = json.dumps(content)
    files: dict[str, str] = {}
    notes: dict[str, str] = {}
    for path, name, note in LAYOUT:
        shown = path.replace("${package}", package)
        values_here = dict(values)
        if name == "pyproject.toml.txt":
            values_here.update({
                "distribution": value_text(distribution), "licence": value_text(licence), "mime_range": mime_range(),
                "description": value_text(f"Entries for MIME in the namespace {namespace}"),
                "authors": ("authors = " + value_text([{"name": each} for each in names]) if names else
                            '# authors = [{ name = "<your name>" }]')})
        files[shown] = fill("plugin", name, values_here)
        notes[shown] = note
    records = f"src/{package}/records"
    files[f"{records}/records.json"] = fill("plugin", "records.json.txt", values)
    notes[f"{records}/records.json"] = "the example record: its sha256 is the data's, filled in"
    files[f"{records}/data/example.json"] = data
    files["LICENSE"] = (template("plugin", "LICENSE.Apache-2.0.txt") if licence == SHIPPED_LICENCE else
                        fill("plugin", "LICENSE.other.txt", {"licence_id": licence}))
    notes["LICENSE"] = (f"the text of {licence}" if licence == SHIPPED_LICENCE else
                        f"names {licence}: put its text here before you publish")
    files["benchmarks/mime.toml"] = manifest_text(
        BUNDLE, f"{distribution}: the example benchmark", content,
        {**defaults.values, "licence": content})
    notes["benchmarks/mime.toml"] = "the record's manifest, copied beside the saved benchmark; `mime check` lists what to write"
    return files, notes


__all__ = ["LAYOUT", "SHIPPED_LICENCE", "plugin_files", "release_branch"]
