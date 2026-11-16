"""Versions are parsed and sorted as versions (F18): an entry's version and a namespace's are numbers joined by dots
(PEP 440), refused at registration otherwise, and "registered at 0.9.0, 0.10.0" lists them in that order, not the
text's (``sorted(["0.9.0", "0.10.0"])`` puts 0.10.0 first).

The controls are the versions MIME's own entries and the fixture lab's carry, which all still register, and a text that
is a version in another spelling (``1``, ``v0.1.0``, ``0.2.0rc1``), which does."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from mime.gradients import GradientStatus
from mime.registry import Delivery, EntryKind, Interaction, Read, RoleKind, register_interaction, registered, versions_of
from mime.testing.toys import LOAD, POSITION
from mime.vocabulary import register_namespace
from mime.vocabulary._versions import version_order, version_problem

register_namespace("versionorder", "0.10.0")


def entry(name: str, version: str) -> type[Interaction]:
    return dataclass(frozen=True, kw_only=True)(type("Versioned", (Interaction,), {
        "name": name, "version": version, "__module__": "versions_test", "differentiability": GradientStatus.EXACT,
        "roles": {"body": RoleKind.BODY}, "reads": {"x": Read("body", POSITION.name)},
        "delivers": {"load": Delivery("body", LOAD.name)}, "__annotations__": {}}))


for held in ("0.9.0", "0.10.0", "0.2.0"):
    register_interaction(entry("versionorder.spring", held))


def test_registered_versions_are_listed_in_the_order_of_versions_not_of_text():
    assert versions_of(EntryKind.INTERACTION, "versionorder.spring") == ["0.2.0", "0.9.0", "0.10.0"]
    listed = [r.version for r in registered(EntryKind.INTERACTION) if r.name == "versionorder.spring"]
    assert listed == ["0.2.0", "0.9.0", "0.10.0"], "the registry's own listing is in the same order"


def test_the_refusal_of_a_version_a_document_names_lists_what_is_registered_in_that_order():
    from mime.saved._resolve import resolve_entry

    found, refusal = resolve_entry("interaction", "versionorder.spring", "0.3.0", "the saved form", [])
    assert found is None and refusal is not None and refusal.code == "load.version"
    assert "at 0.2.0, 0.9.0, 0.10.0 only" in refusal.message, refusal.message


@pytest.mark.parametrize("version", ["banana", "", "0.1.0.x", "v", "a.b.c", None, 1.0])
def test_an_entry_whose_version_is_not_a_version_is_refused_at_registration(version):
    with pytest.raises(TypeError) as error:
        register_interaction(entry("versionorder.bad", version))  # type: ignore[arg-type]
    assert "a version is numbers joined by dots" in str(error.value)


@pytest.mark.parametrize("version", ["1", "v0.1.0", "0.2.0rc1", "1.0.0.post1", "2026.11.16"])
def test_control_the_spellings_of_a_version_that_are_versions_still_register(version):
    register_interaction(entry(f"versionorder.good_{version.replace('.', '_').replace('-', '_')}", version))


def test_a_namespace_whose_version_is_not_a_version_is_refused_and_one_that_is_registers():
    with pytest.raises(ValueError, match="namespace 'versionbad' has version 'banana': a version is numbers joined by dots"):
        register_namespace("versionbad", "banana")
    register_namespace("versionfine", "0.1.0")


def test_control_every_version_of_mime_s_own_entries_and_the_labs_is_a_version():
    held = {r.version for r in registered()}
    assert held and all(version_problem(version) is None for version in held), held


def test_the_sort_key_puts_a_text_that_is_no_version_after_every_version():
    assert sorted(["1.0", "banana", "0.10.0", "0.9.0"], key=version_order) == ["0.9.0", "0.10.0", "1.0", "banana"]
