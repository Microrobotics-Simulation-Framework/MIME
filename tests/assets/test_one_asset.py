"""One ``asset(source)``, its origin decided by the type of ``source`` and never guessed from a string's shape
(DUPLICATES PR B, record item 101), always one ``Asset`` with a record behind it.

- **The three origins.** A string is a record's name (MIME's packaged files are records now, in
  ``mime.data/assets/packaged.json``); a path is a local file, whose record is made when it is read (its sha256 and
  ``NOASSERTION``, so the record alone may not travel); an ``AssetRef`` is a whole record.
- **What it refuses, each with an action:** a string that is a location (``asset.source``: a remote file needs a
  record with its sha256), a string that names no record (``asset.unknown``, its action ``asset(Path(...))``, first
  where the string is a file here), terms or a sha256 given where the source's own are read, and any other type
  (``call.wrong-type``). A setting refuses an origin it does not read: a scene's layer a name (``asset.source``).
- **Unchanged:** every packaged file's name, sha256 and the source the lock writes for it are those MIME pinned before
  the fold (written out here, not read from MIME), so no saved form and no lock moves; a named asset and a file asset
  save in their own document shapes and come back equal; the records keep their tiers (all open).
- **Records on dots:** a record of a source's namespace is found under ``<source>.<name>`` and, this release, under
  the earlier ``<source>_<name>``, keeping its own name (the lab fixture holds one of each); a name outside a source's
  namespace is never re-spelled.
- **Resolvers:** a ``mime.assets`` entry point that names an object is asked for a name in its namespace that nothing
  else holds (``mime.testing.RecordResolver``, declared through a real entry point here), and its file is fetched as any
  record's, from a local mirror. The controls: a resolver that answers another name's record, and one with no
  ``record`` method, are refused (``asset.resolver``) on every run."""

import hashlib
import importlib
import importlib.resources
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from mime import Scene, assets
from mime.assets import Asset, AssetRef, asset, _known, _sources
from mime.assets._asset import lock_source, packaged_path
from mime.errors import MimeError
from mime.saved._document import decode_setting, encode_setting
from mime.scene import Z
from mime.terms import Terms
from mime.testing import RecordResolver

#: MIME's packaged files as ``mime.data.PACKAGED`` pinned them before the fold: the name, the sha256 and the source the
#: lock's ``resolved.assets`` wrote, written out here.
PINNED = {
    "dejongh_mlp_v2": ("ac1fca5c109fa3e2f27520cbd0ee1c0ce5e85ac31a0b1e93e7b8085c68f0a345",
                       "mime.data/mlp_cholesky_weights_v2.npz"),
    "sphere_plane_resistance": ("346bc025d0b56460a20ba98bdfc9023457cba32ca975d44e363585f80a904f5f",
                                "mime.data/sphere_plane_resistance.json"),
    "openmaterica_example_water": ("36add4875e58ade4810d8924cdc5fa7a0dbce28f621c02e4d98fb3516bc92761",
                                   "mime.data/records/water.json"),
    "openmaterica_example_ndfeb_n45": ("40ff82e8a2fb472abcc6cea16bdd594cef78d13c80400d2a94f9926e6b37c4b0",
                                       "mime.data/records/ndfeb_n45.json"),
    "navion_2": ("ea6eeb803b48d73551648e4df08116373a03fed4e871eb2b5c13706f57796b56",
                 "mime.data/emns_calibrations/navion_2.yaml"),
    "octomag": ("d0d92541ea3669bb1c4d4daf4acd36b9d3100f728e6270e2aebac53906cc0ad0",
                "mime.data/emns_calibrations/octomag.yaml"),
    "cardiomag": ("31bd9b234586497275c8abff6ff292ece7ed029be8b6077b027264870facd51c",
                  "mime.data/emns_calibrations/cardiomag.yaml"),
}


def refused(call, *args, **kwargs) -> MimeError:
    with pytest.raises(MimeError) as caught:
        call(*args, **kwargs)
    return caught.value


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MIME_ASSETS_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("MIME_OFFLINE", raising=False)
    monkeypatch.setenv("MIME_ASSET_LOCATIONS", str(tmp_path / "locations.toml"))
    (tmp_path / "locations.toml").write_text("", encoding="utf-8")
    return tmp_path / "cache"


# --- the three origins ---------------------------------------------------------------------------------------------


def test_a_string_is_a_records_name_a_path_a_local_file_and_a_record_a_whole_record(tmp_path: Path):
    named = asset("dejongh_mlp_v2")
    assert (named.origin, named.name, named.sha256, named.path) == ("named", "dejongh_mlp_v2", PINNED["dejongh_mlp_v2"][0],
                                                                    None)
    assert named == Asset("dejongh_mlp_v2", PINNED["dejongh_mlp_v2"][0])
    file = tmp_path / "arm.urdf"
    held = asset(file, sha256="a" * 64, terms=Terms("CC-BY-4.0", "a lab", "results"))
    assert (held.origin, held.name, held.path, held.sha256) == ("file", "arm.urdf", file, "a" * 64)
    record = AssetRef("labq.scan.usda", "b" * 64, 600, "CC0-1.0", ("https://example.org/scan.usda",), role="results")
    whole = asset(record)
    assert (whole.origin, whole.name, whole.sha256, whole.record) == ("record", "labq.scan.usda", "b" * 64, record)
    assert whole.terms == Terms("CC0-1.0", "", "results", ("https://example.org/scan.usda",))


def test_a_local_files_record_is_its_bytes_and_states_no_licence_until_one_is_declared(tmp_path: Path):
    file = tmp_path / "my scan.usda"
    file.write_bytes(b"#usda 1.0\n")
    ref = asset(file).ref()
    assert (ref.name, ref.sha256, ref.size, ref.licence, ref.locations) == (
        "my_scan.usda", hashlib.sha256(b"#usda 1.0\n").hexdigest(), 10, "NOASSERTION", ())
    assert ref.verdict.status == "unknown" and ref.tier == "referenced", "the record alone grants nothing"
    declared = asset(file, terms=Terms("CC-BY-4.0", "a lab", "presentation", ("https://example.org/scan.usda",))).ref()
    assert (declared.licence, declared.credit, declared.role, declared.locations, declared.tier) == (
        "CC-BY-4.0", "a lab", "presentation", ("https://example.org/scan.usda",), "open")


@pytest.mark.parametrize("wrong", [3, b"arm.urdf", None, ["arm.urdf"]])
def test_any_other_type_is_refused_with_the_three_forms(wrong):
    error = refused(asset, wrong)
    assert error.code == "call.wrong-type" and isinstance(error, TypeError)
    assert 'asset("<record name>"), asset(Path("<file>")) or asset(AssetRef(...))' in error.refusals[0].action


@pytest.mark.parametrize("location", ["https://example.org/mesh.stl", "doi:10.5281/zenodo.1/mesh.stl",
                                      "file:///tmp/mesh.stl", "package:labx/marker.usda", "s3://bucket/mesh.stl"])
def test_never_a_url_from_a_string(location):
    error = refused(asset, location)
    assert error.code == "asset.source" and "a remote file needs a record with its sha256" in error.refusals[0].message
    assert f"locations=({location!r},)" in error.refusals[0].action


def test_a_string_that_names_no_record_is_refused_with_the_path_form(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    error = refused(asset, "mesh.stl")
    assert error.code == "asset.unknown" and "write asset(Path('mesh.stl'))" in error.refusals[0].action
    assert "dejongh_mlp_v2" in error.refusals[0].action, "the records a string may name are listed"
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mesh.stl").write_bytes(b"solid\n")
    here = refused(asset, "mesh.stl")
    assert here.code == "asset.unknown" and "is a file here: a string names a record" in here.refusals[0].message
    assert here.refusals[0].action == "write asset(Path('mesh.stl')) for a file on disk"


def test_terms_and_a_sha256_are_refused_where_the_source_has_its_own():
    record = AssetRef("labq.scan.usda", "b" * 64, 600, "CC0-1.0", ("https://example.org/scan.usda",))
    for call in (lambda: asset(record, sha256="b" * 64), lambda: asset(record, terms=Terms("CC0-1.0", "", "results")),
                 lambda: asset("dejongh_mlp_v2", terms=Terms("CC0-1.0", "", "results")),
                 lambda: asset(Path("arm.urdf"), sha256="ABC")):
        assert refused(call).code == "asset.source"


def test_a_scenes_layer_is_a_file_so_a_name_is_refused():
    scene = Scene.new(up=Z, units="m")
    error = refused(scene.reference, asset("dejongh_mlp_v2"), at="/World/UMR")
    assert error.code == "asset.source" and "a scene's layer does not read a record's name" in error.refusals[0].message
    assert refused(scene.reference, "robot.usda", at="/World/UMR").code == "asset.unknown"
    assert scene.references == ()


# --- unchanged: the packaged files, the saved forms, the lock's sources ---------------------------------------------


@pytest.mark.parametrize("name", sorted(PINNED))
def test_each_packaged_file_keeps_its_name_its_pin_and_its_lock_source(name):
    sha256, source = PINNED[name]
    held = asset(name)
    assert held.sha256 == sha256 and lock_source(held) == source
    assert hashlib.sha256(held.read()).hexdigest() == sha256
    assert held.ref().tier == "open" and held.ref().locations == (f"package:mime/{source.removeprefix('mime.data/')}",)


def test_a_packaged_file_is_read_in_place_and_never_copied_into_the_cache(cache: Path):
    assert asset("navion_2").read() and asset("dejongh_mlp_v2").read()
    assert not cache.exists() or not any(cache.rglob("*")), "a packaged file is read from the package, not the cache"


def test_a_name_pinned_to_other_bytes_is_refused_where_it_is_read():
    error = refused(Asset("navion_2", "0" * 64).read)
    assert error.code == "asset.hash-mismatch" and "the installed MIME ships other contents" in str(error)


def test_each_origin_saves_in_its_own_shape_and_comes_back_equal(tmp_path: Path):
    named = asset("octomag")
    assert encode_setting(named, "calibration") == {"asset": "octomag", "sha256": PINNED["octomag"][0]}
    assert decode_setting({"asset": "octomag", "sha256": PINNED["octomag"][0]}) == named
    file = tmp_path / "arm.urdf"
    file.write_bytes(b"<robot name='r'/>\n")
    pin = hashlib.sha256(file.read_bytes()).hexdigest()
    saved = encode_setting(asset(file), "description")
    assert saved == {"file": f"descriptions/{pin[:16]}-arm.urdf", "sha256": pin}
    back = decode_setting(saved, tmp_path)
    assert isinstance(back, Asset) and back.origin == "file" and back.sha256 == pin
    assert back.path == tmp_path / f"descriptions/{pin[:16]}-arm.urdf" and back.name == f"{pin[:16]}-arm.urdf"


# --- records on dots ------------------------------------------------------------------------------------------------


def test_a_record_is_found_under_its_dotted_and_its_earlier_spelling_and_keeps_its_own_name():
    dotted, earlier = assets.reference("labx.marker.usda"), assets.reference("labx_marker.usda")
    assert dotted == earlier and dotted.name == "labx.marker.usda"
    kept, moved = assets.reference("labx_emns_nc.yaml"), assets.reference("labx.emns_nc.yaml")
    assert kept == moved and kept.name == "labx_emns_nc.yaml", "a record spelled the earlier way keeps its name"
    assert asset("labx.emns_nc.yaml").sha256 == kept.sha256


def test_a_name_outside_every_sources_namespace_is_never_respelled():
    assert _sources.spellings("unit_tube_R0.997") == ("unit_tube_R0.997",)
    assert _sources.spellings("labx.marker.usda") == ("labx.marker.usda", "labx_marker.usda")
    assert _sources.spellings("labx_marker.usda") == ("labx_marker.usda", "labx.marker.usda")
    assert refused(assets.reference, "unit.tube_R0.997").code == "asset.unknown"


# --- resolvers ------------------------------------------------------------------------------------------------------

RESOLVER_MODULE = '''
from pathlib import Path
from mime.assets import AssetRef
from mime.testing import RecordResolver

MIRROR = Path(__file__).parent / "mirror"
DATA = (MIRROR / "labr.scan.usda").read_bytes()
RESOLVER = RecordResolver(AssetRef("labr.scan.usda", __import__("hashlib").sha256(DATA).hexdigest(), len(DATA),
                                   "CC0-1.0", (MIRROR.as_uri() + "/",), role="results"))


class Wrong:
    """A planted resolver that answers every name with another name's record."""

    def record(self, name):
        return AssetRef("labw.other.usda", "c" * 64, 1, "CC0-1.0", ())


WRONG = Wrong()
NO_METHOD = object()
'''


def clear_sources() -> None:
    for held in (_sources.shared, _sources.resolvers, _sources._namespaces):
        held.cache_clear()


@pytest.fixture
def installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cache: Path) -> Iterator[Path]:
    """A distribution in ``tmp_path`` whose ``mime.assets`` entry points name three resolvers: ``labr`` (a
    :class:`RecordResolver` of one record, its file in a local mirror), ``labw`` (one that answers another name's
    record) and ``labn`` (an object with no ``record`` method)."""
    root = tmp_path / "site"
    (root / "mirror").mkdir(parents=True)
    (root / "mirror" / "labr.scan.usda").write_bytes(b"#usda 1.0\n(defaultPrim = \"Scan\")\ndef Xform \"Scan\" {}\n")
    (root / "labr_resolver.py").write_text(RESOLVER_MODULE, encoding="utf-8")
    info = root / "labr_resolver-0.1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: labr-resolver\nVersion: 0.1.0\n", encoding="utf-8")
    (info / "entry_points.txt").write_text("[mime.assets]\nlabr = labr_resolver:RESOLVER\nlabw = labr_resolver:WRONG\n"
                                           "labn = labr_resolver:NO_METHOD\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(root))
    importlib.invalidate_caches()
    clear_sources()
    yield root
    clear_sources()


def test_a_resolver_answers_a_name_in_its_namespace_that_nothing_else_holds(installed: Path, cache: Path):
    resolver = importlib.import_module("labr_resolver").RESOLVER
    held = asset("labr.scan.usda")
    ref = held.ref()
    assert held.origin == "named" and held.sha256 == ref.sha256 and ref.locations == ((installed / "mirror").as_uri() + "/",)
    assert "labr.scan.usda" in resolver.asked
    assert "labr.scan.usda" not in _known.known(), "a resolver is asked, never listed"
    with assets.approved([ref]):
        stored = assets.resolve(ref)
    assert stored.read_bytes() == (installed / "mirror" / "labr.scan.usda").read_bytes()
    assert held.read() == stored.read_bytes()
    assert refused(asset, "labr.nothing.usda").code == "asset.unknown"
    assert refused(asset, "labr_scan.usda").code == "asset.unknown", "a resolver is new: it answers dotted names only"


@pytest.mark.parametrize("namespace", ["labw", "labn"])
def test_control_a_resolver_that_answers_wrongly_is_refused(installed: Path, namespace: str):
    error = refused(asset, f"{namespace}.scan.usda")
    assert error.code == "asset.resolver" and f"the asset resolver {namespace!r}" in error.refusals[0].message


def test_a_resolver_named_by_an_object_is_never_read_as_a_folder_of_records(installed: Path):
    assert {source for source, _ in _sources._sources()} >= {"mime", "labx"}
    assert not {"labr", "labw", "labn"} & {source for source, _ in _sources._sources()}
    assert set(_sources.resolvers()) >= {"labr", "labw", "labn"}


def test_the_packaged_records_file_is_well_formed_and_sorted():
    held = json.loads((importlib.resources.files("mime.data") / "assets" / "packaged.json").read_text())
    names = [each["name"] for each in held["assets"]]
    assert names == sorted(PINNED) and all(packaged_path(AssetRef.from_record(each)) for each in held["assets"])
