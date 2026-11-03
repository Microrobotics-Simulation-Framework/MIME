#!/usr/bin/env python3
"""CI script — validate MIME's known anomalies registry.

The registry is in MADDENING's field names where MADDENING has them (``resolution_status``,
``affected_components``, ``affected_versions``, ``resolution_version``, ``verification``), plus three fields
MADDENING lacks, ``planned_resolution``, ``affected_verification_ids`` and ``claims``. This script validates exactly
that, until MADDENING's ``validate_anomaly_registry`` accepts those fields. ``mime docs check`` runs it, handing it the
one collection of the test suite it makes (``--collection``) and the registered components (``--components``), so it
imports no MIME; run alone, it collects the test files the registry names and imports MIME to read the components.

What it checks: the top-level fields; unique IDs with the MIME-ANO- prefix; each entry's required fields; that
``resolution_status`` takes only MADDENING's values; that an entry carries no field outside the known set; that a
``resolved`` entry has a ``resolution_version`` and a ``verification`` list (possibly empty); that every
``verification`` id and every claim's evidence resolves through pytest's collection (``mime.docs._evidence``: a
test that is not collected fails, whatever its file's text holds); that a claim's evidence, its held-out id, and a
``resolved`` entry's verification are not all tests CI does not run (marked ``skip``, ``xfail``, ``slow`` or ``gpu``:
each prints "not run in CI", and is never counted as evidence); that each claim has the claims' shape, its basis
allowing its role; and that ``affected_components`` names each registered component in the controlled form
(``entry: <kind> <name> [<version>][, rung <rung>][; <note>]`` or ``node: <class>[; <note>]``): a legacy spelling
is allowed until an entry or a node of the name it uses exists, and then it fails. It matches names, and does not read
the migration table, so a 0.2 name whose counterpart is registered under another name stays allowed. A ``MIME-NODE-``
id is a permanent id. It does not parse the ``affected_versions`` ranges.
"""

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPOSITORY = Path(__file__).resolve().parent.parent


def _resolver():
    """The evidence resolver, ``mime.docs._evidence``, read from this checkout's source: it needs only the standard
    library and PyYAML, so the script runs without importing MIME when it is handed the registered components
    (``--components``, as ``mime docs check`` hands them over)."""
    name = "mime_docs_evidence_for_the_anomaly_check"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPOSITORY / "src" / "mime" / "docs" / "_evidence.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


_evidence = _resolver()
Collection, claim_problems, collect, evidence_files, lane_counts, resolve = (
    _evidence.Collection, _evidence.claim_problems, _evidence.collect, _evidence.evidence_files,
    _evidence.lane_counts, _evidence.resolve)

# MADDENING's values for resolution_status.
RESOLUTION_STATUSES = ("open", "resolved", "partially_resolved", "wont_fix", "duplicate")

REQUIRED_FIELDS = (
    "anomaly_id", "title", "description",
    "severity", "safety_relevance", "safety_relevance_rationale",
    "resolution_status", "affected_versions",
)
# MADDENING's optional fields, then MIME's own.
OPTIONAL_FIELDS = (
    "affected_components", "workaround", "resolution_version", "verification", "residual_risk", "github_issue",
    "planned_resolution", "affected_verification_ids", "claims",
)
KNOWN_FIELDS = frozenset(REQUIRED_FIELDS + OPTIONAL_FIELDS)

RENAMED_FIELDS = {"status": "resolution_status", "affected_nodes": "affected_components"}

ENTRY_ITEM = re.compile(r"entry: (?P<kind>[a-z_]+) (?P<name>[A-Za-z0-9_.]+)(?: (?P<version>\d+(?:\.\d+)*\S*))?"
                        r"(?:, rung (?P<rung>[a-z0-9_]+))?(?:; (?P<note>.+))?")
NODE_ITEM = re.compile(r"node: (?P<node>[A-Za-z_][A-Za-z0-9_]*)(?:; (?P<note>.+))?")
PERMANENT_ID = re.compile(r"MIME-NODE-\d{3}")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class Components:
    """What a component can be named as: each registered entry's kind and name with its rungs, the names a legacy
    spelling is held to (entries' names and classes, and the node classes they build), and the node classes."""

    entries: dict = field(default_factory=dict)
    names: frozenset = frozenset()
    nodes: frozenset = frozenset()


def components_from(held):
    """The components :func:`mime.docs._evidence.components` wrote as JSON."""
    return Components({tuple(key.split(" ", 1)): frozenset(rungs) for key, rungs in held["entries"].items()},
                      frozenset(held["names"]), frozenset(held["nodes"]))


def registered_components():
    """The components of every registered entry, MIME's and each installed namespace's (this imports MIME)."""
    from mime.docs._export import load_namespaces
    from mime.registry import registered

    load_namespaces()
    return components_from(_evidence.components(registered()))


def legacy_names(item, components):
    """The registered names a legacy spelling uses: an identifier of it (a module path's parts each count), or a
    dotted entry name written whole."""
    words = set(_WORD.findall(item))
    dotted = {name for name in components.names if "." in name
              and re.search(rf"(?<![\w.]){re.escape(name)}(?![\w]|\.\w)", item)}
    return sorted((words & components.names) | dotted)


def component_errors(aid, items, components):
    if not isinstance(items, list) or not all(isinstance(i, str) and i for i in items):
        return [f"{aid}: 'affected_components' must be a list of components"]
    errors = []
    for item in items:
        entry, node = ENTRY_ITEM.fullmatch(item), NODE_ITEM.fullmatch(item)
        if item.startswith("entry:") and entry is None:
            errors.append(f"{aid}: component '{item}' is not 'entry: <kind> <name> [<version>][, rung <rung>]"
                          f"[; <note>]'")
        elif entry is not None:
            key = (entry["kind"], entry["name"])
            if key not in components.entries:
                errors.append(f"{aid}: component '{item}' names no registered entry ({entry['kind']} "
                              f"{entry['name']})")
            elif entry["rung"] and entry["rung"] not in components.entries[key]:
                errors.append(f"{aid}: component '{item}' names the rung {entry['rung']!r}, which "
                              f"{entry['kind']} {entry['name']} does not have")
        elif item.startswith("node:") and node is None:
            errors.append(f"{aid}: component '{item}' is not 'node: <class>[; <note>]'")
        elif node is not None:
            if node["node"] not in components.nodes:
                errors.append(f"{aid}: component '{item}' names no node class a registered entry builds")
        elif not PERMANENT_ID.fullmatch(item):
            used = legacy_names(item, components)
            if used:
                errors.append(f"{aid}: component '{item}' is a legacy spelling of {', '.join(used)}, which is "
                              f"registered now; write it as 'entry: <kind> <name> [<version>][, rung <rung>]"
                              f"[; <note>]' or 'node: <class>[; <note>]'")
    return errors


def evidence_of(entry):
    """Every evidence id ``entry`` names: its verification list and its claims' evidence and held-out ids."""
    found = [i for i in entry.get("verification") or [] if isinstance(i, str)]
    for held in entry.get("claims") or []:
        if isinstance(held, dict):
            found += [i for i in held.get("evidence") or [] if isinstance(i, str)]
            found += [held["held_out"]] if isinstance(held.get("held_out"), str) else []
    return found


def not_run(ids, resolutions):
    """The resolutions of ``ids`` when every one resolves and none is a test CI runs (none counts as evidence)."""
    found = [resolutions[i] for i in ids if i in resolutions]
    if ids and len(found) == len(ids) and all(r.problem is None and not r.counts for r in found):
        return "; ".join(r.text() for r in found)
    return None


def verification_errors(aid, ids, resolutions, status=None):
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        return [f"{aid}: 'verification' must be a list of pytest node ids"]
    errors = [f"{aid}: verification {resolutions[i].text()}" for i in ids
              if i in resolutions and resolutions[i].problem is not None]
    unrun = not_run(ids, resolutions) if status == "resolved" else None
    if unrun is not None:
        errors.append(f"{aid}: a resolved entry's verification is tests CI does not run ({unrun}), which are not "
                      f"evidence; name a test CI runs")
    return errors


def claims_errors(aid, claims, resolutions):
    if not isinstance(claims, list):
        return [f"{aid}: 'claims' must be a list of claims"]
    errors = []
    ids = Counter(held.get("id") for held in claims if isinstance(held, dict))
    errors += [f"{aid}: the claim id {key!r} is used twice" for key, count in ids.items() if count > 1]
    for held in claims:
        errors += claim_problems(held, aid)
        if isinstance(held, dict):
            named = [*(held.get("evidence") or []), *([held["held_out"]] if "held_out" in held else [])]
            errors += [f"{aid}.{held.get('id')}: evidence {resolutions[i].text()}" for i in named
                       if isinstance(i, str) and i in resolutions and resolutions[i].problem is not None]
            evidence = [i for i in held.get("evidence") or [] if isinstance(i, str)]
            for what, ids in (("evidence", evidence), ("held-out evidence", [held.get("held_out")])):
                unrun = not_run([i for i in ids if isinstance(i, str)], resolutions)
                if unrun is not None:
                    errors.append(f"{aid}.{held.get('id')}: its {what} is tests CI does not run ({unrun}), which are "
                                  f"not evidence; a claim rests on a test that runs and passes in a named lane")
    return errors


def entry_errors(aid, entry, resolutions, components):
    errors = []
    for field_name in entry:
        if field_name in RENAMED_FIELDS:
            errors.append(f"{aid}: field '{field_name}' is now '{RENAMED_FIELDS[field_name]}'")
        elif field_name not in KNOWN_FIELDS:
            errors.append(f"{aid}: unknown field '{field_name}'")
    for field_name in REQUIRED_FIELDS:
        if not entry.get(field_name):
            errors.append(f"{aid}: missing required field '{field_name}'")
    status = entry.get("resolution_status")
    if status and status not in RESOLUTION_STATUSES:
        errors.append(f"{aid}: resolution_status '{status}' is not one of {', '.join(RESOLUTION_STATUSES)}")
    if status == "resolved":
        if not entry.get("resolution_version"):
            errors.append(f"{aid}: a resolved entry needs 'resolution_version'")
        if "verification" not in entry:
            errors.append(f"{aid}: a resolved entry needs a 'verification' list (empty if no test exists)")
    if "verification" in entry:
        errors += verification_errors(aid, entry["verification"], resolutions, status)
    if "claims" in entry:
        errors += claims_errors(aid, entry["claims"], resolutions)
    if "affected_components" in entry:
        errors += component_errors(aid, entry["affected_components"], components)
    return errors


@dataclass
class Validation:
    errors: list = field(default_factory=list)
    summary: str = ""


def validate(path, *, prefix="", repository=REPOSITORY, collection=None, components=None, evidence=True):
    """The registry at ``path`` validated, with a one-line summary of what was resolved. ``collection`` is a
    collection of the test suite (one is made of the files the registry names when none is given); ``components``
    the registered components (read from the registry of entries when none are given). Without ``evidence``, no id
    is resolved (``mime docs check --skip-evidence``, whose controls run on copies of the checkout)."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return Validation(["File must contain a YAML mapping"])
    errors = []
    for req in ("schema_version", "generated_date"):
        if req not in data:
            errors.append(f"Missing top-level field: {req}")
    anomalies = [a for a in data.get("anomalies", []) if isinstance(a, dict)]
    named = [i for a in anomalies for i in evidence_of(a)]
    repository = Path(repository)
    if not evidence:
        resolutions = {}
    else:
        if collection is None:
            collection = collect(repository, evidence_files(named))
        resolutions = resolve(named, repository, collection)
    components = components if components is not None else registered_components()
    ids_seen = set()
    for a in anomalies:
        aid = a.get("anomaly_id", "<missing>")
        if aid in ids_seen:
            errors.append(f"Duplicate anomaly_id: {aid}")
        ids_seen.add(aid)
        if prefix and not str(aid).startswith(prefix):
            errors.append(f"{aid}: does not match required prefix '{prefix}'")
        errors += entry_errors(aid, a, resolutions, components)
    claims = sum(len(a.get("claims") or []) for a in anomalies)
    items = [i for a in anomalies for i in a.get("affected_components") or [] if isinstance(i, str)]
    controlled = sum(1 for i in items if ENTRY_ITEM.fullmatch(i) or NODE_ITEM.fullmatch(i))
    resolved = (f"{len(resolutions)} evidence ids collected ({lane_counts(resolutions.values())})" if evidence
                else f"{len(set(named))} evidence ids not resolved (--skip-evidence)")
    summary = (f"{len(anomalies)} anomalies; {resolved}; {claims} claim{'' if claims == 1 else 's'}; "
               f"{controlled} of {len(items)} components in the controlled form, the rest legacy spellings naming "
               f"no registered component")
    return Validation(errors, summary)


def validate_anomaly_registry(path, *, prefix="", repository=REPOSITORY, collection=None, components=None):
    return validate(path, prefix=prefix, repository=repository, collection=collection, components=components).errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--collection", type=Path, help="a collection of the test suite, as mime docs check "
                                                        "writes it (JSON); without it, the files named are collected")
    parser.add_argument("--skip-evidence", action="store_true", help="resolve no evidence id (mime docs check "
                                                                     "--skip-evidence passes it)")
    parser.add_argument("--components", type=Path, help="the registered components, as mime docs check writes them "
                                                        "(JSON); without it, MIME is imported to read them")
    args = parser.parse_args(argv)
    path = "docs/validation/known_anomalies.yaml"
    collection = Collection.from_json(args.collection.read_text(encoding="utf-8")) if args.collection else None
    components = components_from(json.loads(args.components.read_text(encoding="utf-8"))) if args.components else None
    validation = validate(path, prefix="MIME-ANO-", collection=collection, components=components,
                          evidence=not args.skip_evidence)
    if validation.errors:
        for e in validation.errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"OK: anomaly registry at {path} is valid: {validation.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
