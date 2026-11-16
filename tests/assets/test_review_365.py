"""What #365's refuting review found, each with a test that fails without its fix (DUPLICATES PR B).

- **A packaged name is MIME's.** A lock's record, a pin or a script's record that takes the name of a record MIME ships
  under other bytes or terms is refused (``asset.packaged-name``), wherever a record enters: the lock read for
  ``mime assets fetch``, the pins a build keeps, and ``asset(AssetRef(...))``. A record that restates MIME's own changes
  nothing.
- **A resolver owns its namespace.** ``openmaterica.example_water`` asks the ``openmaterica`` resolver; it is never
  respelled onto MIME's ``openmaterica_example_water``, which is still found by its own name. The scaffold refuses the
  first words of MIME's own records as namespaces.
- **A location is never a record's name.** Every spelling of a remote form is ``asset.source``, and its action never
  suggests ``Path(...)``; a ``Path`` that reads as a URL is refused too.
- **A named ``Asset`` carries its sha256**, as it did before the fold.
- **The earlier spelling says so once**, and a resolver that exits is refused (``asset.resolver``) while an interrupt
  propagates."""

import importlib
import json
import types
import warnings
from pathlib import Path

import pytest

from mime import assets
from mime.assets import Asset, AssetRef, asset, _lock, _sources
from mime.errors import MimeError
from mime.saved._pins import keeping_pins
from mime.scaffold._values import namespace_clash, taken_namespaces

NAVION = "ea6eeb803b48d73551648e4df08116373a03fed4e871eb2b5c13706f57796b56"
OTHER = "e" * 64


def refused(call, *args, **kwargs) -> MimeError:
    with pytest.raises(MimeError) as caught:
        call(*args, **kwargs)
    return caught.value


def hostile(name: str = "navion_2", sha256: str = OTHER) -> AssetRef:
    return AssetRef(name, sha256, 10, "Apache-2.0", ("file:///somewhere/a-mirror/",), role="results")


# --- a packaged name is MIME's --------------------------------------------------------------------------------------


def test_a_pin_that_takes_a_packaged_name_for_other_bytes_is_refused_and_one_that_restates_it_is_not():
    error = refused(_sources.pinned([hostile()]).__enter__)
    assert error.code == "asset.packaged-name" and "'navion_2'" in error.refusals[0].message
    with _sources.pinned([assets.reference("navion_2")]):
        assert asset("navion_2").sha256 == NAVION and asset("navion_2").ref().sha256 == NAVION


def test_a_lock_record_that_takes_a_packaged_name_is_refused_where_the_lock_is_read(tmp_path: Path):
    def lock_with(record: AssetRef) -> Path:
        folder = tmp_path / record.name
        folder.mkdir()
        (folder / "mime.lock").write_text(json.dumps({"assets": [record.to_record()]}), encoding="utf-8")
        return folder

    assert refused(_lock.folder_assets, lock_with(hostile())).code == "asset.packaged-name"
    assert [each.name for each in _lock.folder_assets(lock_with(hostile("labx.other.bin")))] == ["labx.other.bin"]


def test_a_build_keeping_a_record_that_takes_a_packaged_name_is_refused():
    brought = types.SimpleNamespace(assets=(hostile(),))
    error = refused(keeping_pins(None, brought).__enter__)
    assert error.code == "asset.packaged-name"


def test_a_script_declaring_a_record_under_a_packaged_name_is_refused():
    assert refused(asset, hostile()).code == "asset.packaged-name"
    assert asset(assets.reference("navion_2")).sha256 == NAVION


# --- a resolver owns its namespace ----------------------------------------------------------------------------------

RESOLVERS = '''
from mime.assets import AssetRef


class Namespace:
    def __init__(self):
        self.asked = []

    def record(self, name):
        self.asked.append(name)
        return AssetRef(name, "c" * 64, 3, "CC0-1.0", ("https://example.org/w.json",))


class Exits:
    def record(self, name):
        raise SystemExit(3)


class Interrupted:
    def record(self, name):
        raise KeyboardInterrupt


OPENMATERICA, DEJONGH, EXITS, INTERRUPTED = Namespace(), Namespace(), Exits(), Interrupted()
'''


def clear_sources() -> None:
    for held in (_sources.shared, _sources.resolvers, _sources._namespaces):
        held.cache_clear()


@pytest.fixture
def owning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Resolvers that own ``openmaterica`` and ``dejongh`` (the first words of MIME's own records), and two that exit."""
    root = tmp_path / "site"
    root.mkdir()
    (root / "owning_resolvers.py").write_text(RESOLVERS, encoding="utf-8")
    info = root / "owning_resolvers-0.1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: owning-resolvers\nVersion: 0.1.0\n", encoding="utf-8")
    (info / "entry_points.txt").write_text(
        "[mime.assets]\nopenmaterica = owning_resolvers:OPENMATERICA\ndejongh = owning_resolvers:DEJONGH\n"
        "labexit = owning_resolvers:EXITS\nlabstop = owning_resolvers:INTERRUPTED\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(root))
    importlib.invalidate_caches()
    clear_sources()
    module = importlib.import_module("owning_resolvers")
    module.OPENMATERICA.asked, module.DEJONGH.asked = [], []
    yield module
    clear_sources()


def test_a_resolver_owned_namespace_is_asked_first_and_never_respelled_onto_mimes_own_name(owning):
    held = asset("openmaterica.example_water")
    assert held.ref().name == "openmaterica.example_water" and held.sha256 == "c" * 64
    assert set(owning.OPENMATERICA.asked) == {"openmaterica.example_water"}
    other = asset("dejongh.mlp_v2")
    assert other.sha256 == "c" * 64 and owning.DEJONGH.asked[0] == "dejongh.mlp_v2"
    assert _sources.spellings("openmaterica_example_water") == ("openmaterica_example_water",)
    assert _sources.spellings("openmaterica.example_water") == ("openmaterica.example_water",)


def test_mimes_own_name_is_still_found_by_its_own_name_where_a_resolver_owns_the_first_word(owning):
    water = asset("openmaterica_example_water")
    assert water.ref().name == "openmaterica_example_water" and water.sha256 == \
        "36add4875e58ade4810d8924cdc5fa7a0dbce28f621c02e4d98fb3516bc92761"
    assert owning.OPENMATERICA.asked == []


def test_a_resolver_that_exits_is_refused_and_an_interrupt_propagates(owning):
    error = refused(asset, "labexit.x")
    assert error.code == "asset.resolver" and "SystemExit" in error.refusals[0].message
    with pytest.raises(KeyboardInterrupt):
        asset("labstop.x")


@pytest.mark.parametrize("word", ["openmaterica", "dejongh", "unit", "navion", "octomag", "cardiomag"])
def test_the_scaffold_refuses_the_first_word_of_a_record_mime_ships(word: str):
    assert word in taken_namespaces() and namespace_clash(word) is not None


# --- locations are never names ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("written", ["DOI:10.5281/zenodo.1/x", "HTTP:x", "Https:x", "s3:b/k", "//host/x",
                                     "\\\\host\\share\\x", "https://example.org/x.usda", "doi:10.5281/zenodo.1/x"])
def test_every_spelling_of_a_remote_form_is_asset_source_and_never_suggests_a_path(written: str):
    error = refused(asset, written)
    assert error.code == "asset.source", written
    assert "Path(" not in error.refusals[0].action, error.refusals[0].action


def test_a_path_that_reads_as_a_url_is_refused():
    error = refused(asset, Path("https://example.org/x.usda"))
    assert error.code == "asset.source" and "Path(" not in error.refusals[0].action
    assert refused(asset, "arm.urdf").code == "asset.unknown", "a plain name stays unknown, with Path(...) as its action"


# --- a named Asset carries its sha256 --------------------------------------------------------------------------------


@pytest.mark.parametrize("sha256", [None, "", "abc"])
def test_a_named_asset_without_a_sha256_is_refused(sha256):
    assert refused(Asset, "navion_2", sha256).code == "asset.source"
    assert Asset("navion_2", NAVION).sha256 == NAVION


# --- the earlier spelling says so once ---------------------------------------------------------------------------------


def test_the_earlier_spelling_of_a_record_says_once_which_to_write():
    _sources._WARNED.discard("labx_marker.usda")
    with pytest.warns(FutureWarning, match=r"write 'labx.marker.usda'"):
        assets.reference("labx_marker.usda")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assets.reference("labx_marker.usda")
        assets.reference("labx.marker.usda")
        assets.reference("labx_emns_nc.yaml")
