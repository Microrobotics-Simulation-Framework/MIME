"""Saved form 14's ``blocks`` key: each block the experiment uses, by the use's name, written only where one is used and
inside the input hash. Its writer (``exp.use``) and the reading of a block used by reference come later; until then a
block used as ``copied`` (its declarations in the document) loads, and one used as ``referenced`` (its declarations in
its container) is refused by its own code, ``load.referenced-block``, never as a shape. The uses here are written
through the carrier ``exp.use`` will set, ``RunSettings.blocks``, and drawn from ``MIME_SEED``.

**The controls**, each run by ``detection_rate.py`` on a probe (``MIME_PLANT_BLOCKS``): ``dropped`` loses the key on
the way back in, so the load's rebuilt document holds none and its input hash is not the lock's; ``unchecked`` removes
the refusal of a referenced use, which then loads as if copied; ``unstepped`` writes the folder as form 13, which
version 13's reader then refuses as a shape, never as newer."""

import hashlib
import os
import random
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_form11 import codes_of, reader_of, restated
from test_form14 import NEWER, own_cache, scanned13

from mime.build import Attachment, ThingDeclaration, entry_id
from mime.experiment import Experiment
from mime.experiment import _experiment
from mime.saved import DOCUMENT_FILE, LoadRefused, read_folder
from mime.saved import _reconstruct, _versions
from mime.saved._versions import BLOCKS_KEY, HASH_ONLY_ADDED

#: The form this MIME read before the step.
FORM13_READS = 13
LICENCES = ("CC-BY-4.0", "CC-BY-NC-4.0", "CC-BY-SA-4.0", "CC0-1.0", "MIT", "Apache-2.0")
USE_NAMES = ("probe_unit", "vessel", "coil_rig", "transducer")
BLOCK_IDS = ("labx.demo_probe", "labx.tube_network", "laby.coil_rig")
REFERENCED = "load.referenced-block"


def plant() -> str:
    return os.environ.get("MIME_PLANT_BLOCKS", "")


def seed() -> int:
    return int(os.environ.get("MIME_SEED", "0"))


def toy() -> Experiment:
    from mime.testing._graph import declare_in_order
    from mime.testing.toys._experiments import ADD_PROBE, ATTACH_SPRING, toy_experiment

    return declare_in_order(lambda: toy_experiment("blocks"), (ADD_PROBE, ATTACH_SPRING), None)


def provided(experiment: Experiment) -> dict[str, list[str]]:
    """The names a block could have provided to ``experiment``, as its document holds them."""
    declarations = experiment._declarations  # noqa: SLF001
    return {"things": [held.thing.name for held in declarations if isinstance(held, ThingDeclaration)],
            "interactions": [entry_id(type(held.law), held.bindings) for held in declarations
                             if isinstance(held, Attachment)]}


def drawn_uses(rng: random.Random, tiers: tuple[str, ...], names: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
    """One use for each of ``tiers``, under distinct names, each entry drawn as ``exp.use`` would write it."""
    uses: dict[str, dict[str, Any]] = {}
    for name, tier in zip(rng.sample(USE_NAMES, len(tiers)), tiers, strict=True):
        use: dict[str, Any] = {
            "id": rng.choice(BLOCK_IDS), "version": f"{rng.randint(0, 3)}.{rng.randint(0, 9)}.{rng.randint(0, 20)}",
            "content": hashlib.sha256(rng.randbytes(32)).hexdigest(), "tier": tier, "licence": rng.choice(LICENCES),
            "sockets": {socket: rng.choice(("world", "probe")) for socket in
                        rng.sample(("medium", "subject", "base"), rng.randint(0, 3))},
            "names": {kind: held for kind, held in names.items() if rng.random() < 0.7},
            "overrides": sorted(rng.sample(("probe.mass", "probe.radius", "probe.inertia"), rng.randint(0, 2)))}
        if rng.random() < 0.5:
            use["doi"] = f"10.5281/zenodo.{rng.randint(10**6, 10**7)}"
        uses[name] = use
    return uses


def saved_with(out: Path, tiers: tuple[str, ...]) -> tuple[Path, dict[str, dict[str, Any]]]:
    """The toy saved into ``out`` using a block for each of ``tiers``, drawn from ``MIME_SEED``, and its uses."""
    experiment = toy()
    uses = drawn_uses(random.Random(seed()), tiers, provided(experiment))
    experiment._settings = replace(experiment._settings, blocks=uses)  # noqa: SLF001
    with own_cache(out.parent):
        experiment.build().save(out)
    return out, uses


@contextmanager
def planted(value: str) -> Iterator[None]:
    """``dropped``: the load reads the document's uses back as none. ``unchecked``: a referenced use is not refused."""
    read = _experiment.declarations_from

    def dropped(*args: Any) -> Any:
        settings, declarations = read(*args)
        return replace(settings, blocks=None), declarations

    with pytest.MonkeyPatch.context() as patch:
        if plant() == value == "dropped":
            patch.setattr(_experiment, "declarations_from", dropped)
        if plant() == value == "unchecked":
            patch.setattr(_reconstruct, "referenced_block_refusals", lambda document: [])
        yield


def test_a_copied_block_is_written_into_the_document_and_loads_with_its_input_hash_reproduced(tmp_path: Path):
    folder, uses = saved_with(tmp_path / "copied", ("copied",) * random.Random(seed()).randint(1, 2))
    document = yaml.safe_load((folder / DOCUMENT_FILE).read_text(encoding="utf-8"))
    assert document[BLOCKS_KEY] == uses and list(document)[:4] == ["schema", "mime", "experiment", BLOCKS_KEY]
    with planted("dropped"), own_cache(tmp_path):
        loaded = Experiment.load(folder)
    assert loaded.load_report is not None and "input hash: reproduced exactly" in loaded.load_report.text()
    assert loaded._settings.blocks == uses  # noqa: SLF001


def test_a_use_edited_after_its_lock_was_written_is_refused_by_the_input_hash(tmp_path: Path):
    """The key is inside the input hash: one use's content hash changed by hand is refused as any edit is."""
    folder, uses = saved_with(tmp_path / "edited", ("copied",))
    document = yaml.safe_load((folder / DOCUMENT_FILE).read_text(encoding="utf-8"))
    (name,) = uses
    document[BLOCKS_KEY][name]["content"] = hashlib.sha256(b"other content").hexdigest()
    (folder / DOCUMENT_FILE).write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with own_cache(tmp_path), pytest.raises(LoadRefused) as refused:
        Experiment.load(folder)
    assert refused.value.codes() == ["load.input-hash"]


def test_a_referenced_block_is_refused_by_its_own_code_naming_each_use_and_never_as_a_shape(tmp_path: Path):
    tiers = random.Random(seed()).choice((("referenced",), ("referenced", "copied"), ("copied", "referenced"),
                                          ("referenced", "referenced")))
    folder, uses = saved_with(tmp_path / "referenced", tiers)
    assert codes_of(folder) == [], "the folder is read: show, pack and card refuse nothing"
    held = sorted(name for name, use in uses.items() if use["tier"] == "referenced")
    for load in (lambda: Experiment.load(folder), lambda: Experiment.from_document(folder / DOCUMENT_FILE)):
        with planted("unchecked"), own_cache(tmp_path), pytest.raises(LoadRefused) as refused:
            load()
        assert refused.value.codes() == [REFERENCED] * len(held)
        for name, message in zip(held, (each.message for each in refused.value.refusals), strict=True):
            use = uses[name]
            assert message.startswith(f"the document uses the block {name!r} ({use['id']} {use['version']}) by "
                                      f"reference") and "does not yet read a referenced block" in message


def test_version_thirteens_reader_refuses_a_document_using_a_block_as_newer_each_file_once(tmp_path: Path):
    folder, _ = saved_with(tmp_path / "used", (random.Random(seed()).choice(("copied", "referenced")),))
    assert codes_of(folder) == [], "this MIME reads it"
    read = restated(folder, tmp_path / "unstepped", FORM13_READS) if plant() == "unstepped" else folder
    with reader_of(FORM13_READS):
        assert codes_of(read) == [NEWER, NEWER]


def test_control_the_form_left_unstepped_is_refused_by_version_thirteens_reader_as_a_shape_never_as_newer(
        tmp_path: Path):
    folder, _ = saved_with(tmp_path / "used", ("copied",))
    unstepped = restated(folder, tmp_path / "unstepped", FORM13_READS)
    with reader_of(FORM13_READS), pytest.raises(LoadRefused) as refused:
        read_folder(unstepped)
    assert NEWER not in refused.value.codes() and "load.document-shape" in refused.value.codes()


def test_a_form_thirteen_document_holding_blocks_is_refused_and_written_back_without_it(tmp_path: Path):
    """A form-13 document that holds the key is refused by the version that added it, never read as using a block,
    and a rebuilt document written back into form 13 has it taken off; with the step's check removed, it is read."""
    _, uses = saved_with(tmp_path / "drawn", ("copied",))
    folder = scanned13(tmp_path / "scanned")
    document = yaml.safe_load((folder / DOCUMENT_FILE).read_text(encoding="utf-8"))
    (folder / DOCUMENT_FILE).write_text(yaml.safe_dump({**document, BLOCKS_KEY: uses}), encoding="utf-8")
    with pytest.raises(LoadRefused) as refused:
        read_folder(folder)
    assert refused.value.codes()[0] == "load.document-shape"
    assert f"document holds {BLOCKS_KEY}, which version {HASH_ONLY_ADDED} of the saved form added" in \
        str(refused.value)
    assert _versions.document_in_form({**document, BLOCKS_KEY: uses}, FORM13_READS) == document
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_versions, "STEP_14_DOCUMENT_KEYS", ())
        assert read_folder(folder).version == FORM13_READS
        assert _versions.document_in_form({**document, BLOCKS_KEY: uses}, FORM13_READS) != document
