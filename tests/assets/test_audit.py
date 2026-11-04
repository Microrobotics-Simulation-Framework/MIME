"""The guard (design #229 item 9): ``mime assets audit`` finds, in a wheel, an sdist, an archive or a tree, any file
whose bytes are a referenced record's, a mesh baked from one in the local cache, and a mesh no record names; a mesh
whose record is open passes. Every test mesh is made here (a synthetic STL, OBJ, PLY, glTF and USD, and a layer MIME's
own baking writes): no upstream mesh is ever downloaded into a test.

Its reach, each with a planted copy it finds: **formats by their bytes**, renamed to ``.bin`` (binary and ASCII STL,
USD's crate and text, OBJ, PLY, glTF and its binary, a layer MIME's baking writes); **compressed copies** (gzip, bzip2,
xz); **archives one inside another**, by content (a zip in a tar in a gzip in a wheel, a ``.tar.gz`` in a zip, a zip
named ``.dat``), to its stated depth, a deeper one refused; **base64 blocks** in a text file (a file of base64, a data
URI, lines broken at 76) at its stated floor and over; **no directory skipped by name** (a ``.git`` below the top of a
tree, and anywhere in an archive), the checkout's own ``.git`` at the top left out and said. **Its limits refuse, never
skip**: a file it must look inside over its size, a target that expands past its total or holds more files than its
count, and an archive it cannot read, each ``asset.audit``; a large file of any other kind is hashed as it is read, and
found.

The controls, each failing on every run: a guard planted to ignore the records inside its target misses a referenced
copy that only the target's own manifest names; a guard planted to look at file names alone misses an STL renamed to
``.bin``, and every other format by its bytes; a guard that opens no compressed stream misses each compressed copy; a
guard that decodes no base64 misses each block; a guard that prunes every ``.git`` misses the one below the top."""

import base64
import bz2
import gzip
import hashlib
import io
import json
import lzma
import struct
import tarfile
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from mime import assets
from mime.assets import _audit
from mime.cli import main

RUNS = 20


def stl(triangles: int, seed: int = 0) -> bytes:
    """A binary STL file of ``triangles`` facets, made here."""
    body = b"".join(struct.pack("<12fH", *(float((seed + i + k) % 7) for k in range(12)), 0) for i in range(triangles))
    return b"made by a test".ljust(80, b"\0") + struct.pack("<I", triangles) + body


def manifest(name: str, data: bytes, licence: str, redistribution: str = "not_permitted") -> dict[str, object]:
    record: dict[str, object] = {"name": f"{name}.stl", "sha256": hashlib.sha256(data).hexdigest(), "size": len(data),
                                 "licence": licence, "locations": [f"https://example.org/{name}.stl"],
                                 "role": "presentation"}
    held: dict[str, object] = {"format": "mime.asset_manifest/1", "name": name, "title": name, "credit": "a maker",
                               "notice": "terms", "archives": [], "assets": [record],
                               "files": {f"meshes/{name}.stl": {"asset": f"{name}.stl"}}}
    if licence.startswith("LicenseRef-"):
        held["licences"] = {licence: {"title": "a maker's licence", "url": "https://example.org/licence",
                                      "redistribution": redistribution}}
    return held


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MIME_ASSETS_DIR", str(tmp_path / "cache"))
    return tmp_path / "cache"


def tree(root: Path, files: dict[str, bytes]) -> Path:
    for name, data in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
    return root


def kinds(report: assets.AuditReport) -> list[tuple[str, str]]:
    return sorted((finding.path, finding.kind) for finding in report.findings)


def test_a_referenced_copy_and_an_unrecorded_mesh_are_found_and_an_open_recorded_mesh_passes(cache, tmp_path):
    closed, loose, open_ = stl(10, 1), stl(11, 2), stl(12, 3)
    root = tree(tmp_path / "plugin", {
        "plugin/records/closed.json": json.dumps(manifest("closed", closed, "LicenseRef-Maker")).encode(),
        "plugin/records/open.json": json.dumps(manifest("open", open_, "CC-BY-4.0")).encode(),
        "plugin/data/closed.stl": closed, "plugin/data/loose.stl": loose, "plugin/data/open.stl": open_,
        "plugin/__init__.py": b""})
    report = assets.audit(root)
    assert kinds(report) == [("plugin/data/closed.stl", "referenced"), ("plugin/data/loose.stl", "unrecorded mesh")]


def test_a_wheel_and_a_tar_are_read_and_an_archive_inside_one_level_down(cache, tmp_path):
    closed = stl(9, 4)
    held = json.dumps(manifest("closed", closed, "LicenseRef-Maker")).encode()
    wheel = tmp_path / "plugin-0.1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("plugin/records.json", held)
        inner = tmp_path / "inner.zip"
        with zipfile.ZipFile(inner, "w") as nested:
            nested.writestr("x/closed.dat", closed)
        archive.writestr("plugin/bundle.mimebundle", inner.read_bytes())
    assert kinds(assets.audit(wheel)) == [("plugin/bundle.mimebundle!x/closed.dat", "referenced")]
    sdist = tmp_path / "plugin-0.1.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        for name, data in (("plugin-0.1/records.json", held), ("plugin-0.1/meshes/c.stl", closed)):
            source = tmp_path / "member"
            source.write_bytes(data)
            archive.add(source, arcname=name)
    assert kinds(assets.audit(sdist)) == [("plugin-0.1/meshes/c.stl", "referenced")]


def test_a_mesh_baked_into_the_local_cache_from_a_referenced_source_is_found_by_hash(cache, tmp_path):
    baked = b"PXR-USDC" + bytes(range(256)) * 8
    folder = cache / "visuals"
    folder.mkdir(parents=True)
    (folder / "k.json").write_text(json.dumps({"format": "mime.baked_mesh/3", "set": "maker", "file": "m/a.stl",
                                               "output": {"sha256": hashlib.sha256(baked).hexdigest()},
                                               "tier": "referenced"}), encoding="utf-8")
    root = tree(tmp_path / "run", {"experiment/scene/copied.layer": baked})
    assert kinds(assets.audit(root)) == [("experiment/scene/copied.layer", "derived")]
    record = json.loads((folder / "k.json").read_text(encoding="utf-8"))
    (folder / "k.json").write_text(json.dumps({**record, "format": "mime.baked_mesh/2", "tier": "no-redistribution"}),
                                   encoding="utf-8")
    assert kinds(assets.audit(root)) == [("experiment/scene/copied.layer", "derived")], "an earlier MIME's bake"
    assert assets.audit(root, cache=False).findings[0].kind == "unrecorded mesh", "a USD crate is a mesh format"


def test_the_ar4s_records_name_its_meshes_and_a_made_copy_under_one_of_their_hashes_is_found(cache, tmp_path,
                                                                                          monkeypatch):
    """The AR4's manifest counts wherever it is: inside the target, or given. A file is held to it by hash, so a test
    plants a made file and a record of the same made bytes, never one of Annin Robotics' meshes."""
    from chain_cases import AR4_VISUALS

    made = stl(5, 9)
    held = json.loads(AR4_VISUALS.read_text(encoding="utf-8"))
    held["assets"][0]["sha256"], held["assets"][0]["size"] = hashlib.sha256(made).hexdigest(), len(made)
    planted = tmp_path / "visuals.json"
    planted.write_text(json.dumps(held), encoding="utf-8")
    root = tree(tmp_path / "tree", {"pkg/thing.bin": made})
    assert kinds(assets.audit(root)) == [("pkg/thing.bin", "unrecorded mesh")]
    assert kinds(assets.audit(root, manifests=[planted])) == [("pkg/thing.bin", "referenced")]


def test_the_command_exits_one_on_a_finding_and_zero_on_none(cache, tmp_path, capsys):
    clean = tree(tmp_path / "clean", {"a.py": b"x = 1\n"})
    dirty = tree(tmp_path / "dirty", {"m.stl": stl(3)})
    assert main(["assets", "audit", str(clean)]) == 0
    assert main(["assets", "audit", str(clean), str(dirty)]) == 1
    out = capsys.readouterr().out
    assert "m.stl: unrecorded mesh, a binary STL file no record names" in out and "no file that may not be" in out


@contextmanager
def target_records_ignored() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_audit, "_inside", lambda name, data: [])
        yield


@contextmanager
def names_only() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_audit, "mesh_format", lambda name, data: "STL" if name.lower().endswith(".stl") else None)
        yield


def own_manifest_finds_its_copy(tmp_path: Path) -> None:
    closed = stl(8, 5)
    root = tree(tmp_path, {"r.json": json.dumps(manifest("closed", closed, "LicenseRef-Maker")).encode(),
                           "data/x.bin": closed})
    assert kinds(assets.audit(root, cache=False)) == [("data/x.bin", "referenced")]


def renamed_stl_is_found(tmp_path: Path) -> None:
    root = tree(tmp_path, {"data/renamed.bin": stl(6, 7)})
    assert kinds(assets.audit(root, cache=False)) == [("data/renamed.bin", "unrecorded mesh")]


@pytest.mark.parametrize("scenario,plant", [(own_manifest_finds_its_copy, target_records_ignored),
                                            (renamed_stl_is_found, names_only)])
def test_control_each_plant_is_caught_every_run(tmp_path, scenario, plant):
    scenario(tmp_path / "unplanted")
    caught = 0
    for run in range(RUNS):
        with plant():
            try:
                scenario(tmp_path / str(run))
            except AssertionError:
                caught += 1
    assert caught == RUNS


def test_the_repository_tree_holds_no_file_that_may_not_be_redistributed(cache, tmp_path):
    """The release check: the committed tree as ``git archive`` writes it (what a GitHub release's source archive and
    Zenodo's snapshot of a tag hold) is clean."""
    import subprocess

    root = Path(__file__).resolve().parents[2]
    archive = tmp_path / "tree.tar"
    subprocess.run(["git", "archive", "--format=tar", "-o", str(archive), "HEAD"], cwd=root, check=True,
                   capture_output=True)
    report = assets.audit(archive, cache=False)
    assert report.files > 500 and report.findings == ()


def test_a_target_that_is_not_there_is_refused(cache, tmp_path):
    from mime.build import BuildRefused

    with pytest.raises(BuildRefused) as refused:
        assets.audit(tmp_path / "nothing.whl")
    assert refused.value.codes() == ["asset.audit"]


# --- formats by their bytes ---------------------------------------------------------------------------------------


def ascii_stl() -> bytes:
    facets = "".join(f"facet normal 0 0 1\n outer loop\n vertex 0 0 {i}\n vertex 1 0 {i}\n vertex 0 1 {i}\n"
                     f" endloop\nendfacet\n" for i in range(4))
    return f"solid made\n{facets}endsolid made\n".encode()


def obj() -> bytes:
    return b"# made\nv 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\nf 1 3 4\n"


def ply() -> bytes:
    return (b"ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
            b"element face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n")


def gltf() -> bytes:
    return json.dumps({"asset": {"version": "2.0"}, "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
                       "accessors": [{"count": 3, "type": "VEC3", "componentType": 5126}]}).encode()


def glb() -> bytes:
    body = gltf().ljust(len(gltf()) + (-len(gltf()) % 4), b" ")
    chunk = struct.pack("<I4s", len(body), b"JSON") + body
    return struct.pack("<4sII", b"glTF", 2, 12 + len(chunk)) + chunk


def usd_text() -> bytes:
    return (b'#usda 1.0\n(\n    defaultPrim = "m"\n)\n\ndef Mesh "m"\n{\n'
            b'    point3f[] points = [(0, 0, 0), (1, 0, 0)]\n}\n')


def baked_layer(tmp_path: Path) -> bytes:
    """A layer as MIME's baking writes it (``mime.viz._meshes.write_mesh``, a USD crate), of a made mesh."""
    import numpy as np

    from mime.viz._meshes import Mesh, write_mesh

    mesh = Mesh(np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]]), np.array([[0, 1, 2]]))
    return write_mesh(tmp_path / "baked.usdc", mesh, "mesh").read_bytes()


FORMATS = {"binary STL": lambda tmp: stl(7, 11), "ASCII STL": lambda tmp: ascii_stl(),
           "USD crate": baked_layer, "USD text": lambda tmp: usd_text(), "OBJ": lambda tmp: obj(),
           "PLY": lambda tmp: ply(), "glTF": lambda tmp: gltf(), "glTF binary": lambda tmp: glb()}


@pytest.mark.parametrize("form", sorted(FORMATS))
def test_a_mesh_or_scene_format_is_read_from_its_bytes_whatever_its_name(cache, tmp_path, form):
    data = FORMATS[form](tmp_path)
    assert _audit.mesh_format("blob.bin", data) == form
    root = tree(tmp_path / "t", {"pkg/blob.bin": data, "pkg/notes.txt": b"no mesh here\n"})
    assert kinds(assets.audit(root, cache=False)) == [("pkg/blob.bin", "unrecorded mesh")]


def test_mimes_own_baked_layer_is_a_mesh_by_its_bytes(cache, tmp_path):
    assert _audit.mesh_format("layer", baked_layer(tmp_path)) == "USD crate"


@pytest.mark.parametrize("form", sorted(FORMATS))
def test_control_a_guard_that_reads_names_alone_misses_each_format_every_run(tmp_path, form):
    caught = 0
    for run in range(RUNS):
        with names_only():
            try:
                test_a_mesh_or_scene_format_is_read_from_its_bytes_whatever_its_name(None, tmp_path / str(run), form)
            except AssertionError:
                caught += 1
    assert caught == RUNS


# --- compressed, nested, encoded ----------------------------------------------------------------------------------


CLOSED = stl(12, 21)
COMPRESSIONS = {"gzip": gzip.compress, "bzip2": bz2.compress, "xz": lzma.compress}


def closed_tree(tmp_path: Path, files: dict[str, bytes]) -> Path:
    return tree(tmp_path, {"r.json": json.dumps(manifest("closed", CLOSED, "LicenseRef-Maker")).encode(), **files})


@pytest.mark.parametrize("kind", sorted(COMPRESSIONS))
def test_a_compressed_copy_is_found_inside_its_stream(cache, tmp_path, kind):
    suffix = {"gzip": ".gz", "bzip2": ".bz2", "xz": ".xz"}[kind]
    root = closed_tree(tmp_path / "t", {f"pkg/closed.stl{suffix}": COMPRESSIONS[kind](CLOSED),
                                        "pkg/renamed.dat": COMPRESSIONS[kind](CLOSED)})
    assert kinds(assets.audit(root, cache=False)) == [
        (f"pkg/closed.stl{suffix}!closed.stl", "referenced"), (f"pkg/renamed.dat!({kind} stream)", "referenced")]


def zipped(files: dict[str, bytes]) -> bytes:
    held = io.BytesIO()
    with zipfile.ZipFile(held, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return held.getvalue()


def tarred(files: dict[str, bytes], compress: bool = True) -> bytes:
    held = io.BytesIO()
    with tarfile.open(fileobj=held, mode="w:gz" if compress else "w") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return held.getvalue()


def test_archives_one_inside_another_are_read_by_content_to_the_stated_depth(cache, tmp_path):
    record = json.dumps(manifest("closed", CLOSED, "LicenseRef-Maker")).encode()
    in_wheel = zipped({"pkg/r.json": record, "pkg/deep.tar.gz": tarred({"inner/x.zip": zipped({"m.stl": CLOSED})}),
                       "pkg/named.dat": zipped({"m.bin": CLOSED})})
    wheel = tmp_path / "plugin-0.1-py3-none-any.whl"
    wheel.write_bytes(in_wheel)
    assert kinds(assets.audit(wheel, cache=False)) == [
        ("pkg/deep.tar.gz!inner/x.zip!m.stl", "referenced"), ("pkg/named.dat!m.bin", "referenced")]
    sdist = tmp_path / "plugin-0.1.tar.gz"
    sdist.write_bytes(tarred({"plugin-0.1/r.json": record, "plugin-0.1/data.zip": zipped({"t.tar": tarred(
        {"m.stl": CLOSED}, compress=False)})}))
    assert kinds(assets.audit(sdist, cache=False)) == [("plugin-0.1/data.zip!t.tar!m.stl", "referenced")]


def test_an_archive_deeper_than_the_stated_depth_is_refused(cache, tmp_path):
    from mime.build import BuildRefused

    held = CLOSED
    for _ in range(_audit.DEPTH_LIMIT + 1):
        held = zipped({"inner": held})
    root = tree(tmp_path / "t", {"deep.zip": held})
    with pytest.raises(BuildRefused) as refused:
        assets.audit(root, cache=False)
    assert refused.value.codes() == ["asset.audit"] and "layers deep" in str(refused.value)
    shallower = tree(tmp_path / "s", {"deep.zip": zipped({"inner": CLOSED})})
    assert kinds(assets.audit(shallower, cache=False)) == [("deep.zip!inner", "unrecorded mesh")]


def test_a_base64_block_in_a_text_file_is_read_at_the_stated_floor_and_over(cache, tmp_path):
    encoded = base64.b64encode(CLOSED)
    wrapped = b"\n".join(encoded[i:i + 76] for i in range(0, len(encoded), 76))
    small = base64.b64encode(CLOSED[:_audit.BASE64_MIN_BYTES - 1])
    root = closed_tree(tmp_path / "t", {
        "pkg/closed.b64": encoded, "pkg/page.html": b'<img src="data:model/stl;base64,' + encoded + b'">',
        "pkg/mail.txt": b"Content-Transfer-Encoding: base64\n\n" + wrapped + b"\n",
        "pkg/safe.json": json.dumps({"mesh": base64.urlsafe_b64encode(CLOSED).decode()}).encode(),
        "pkg/small.txt": small})
    found = kinds(assets.audit(root, cache=False))
    assert [path.partition("#")[0] for path, kind in found if kind == "referenced"] == [
        "pkg/closed.b64", "pkg/mail.txt", "pkg/page.html", "pkg/safe.json"]
    assert all("#base64@" in path for path, _ in found) and len(found) == 4, found


def test_no_directory_is_skipped_by_name_but_a_checkouts_own_git(cache, tmp_path):
    root = closed_tree(tmp_path / "checkout", {"sub/.git/objects/m.stl": CLOSED, ".git/objects/history.stl": CLOSED,
                                                "pkg/vendored.zip": zipped({".git/m.stl": CLOSED})})
    report = assets.audit(root, cache=False)
    assert kinds(report) == [("pkg/vendored.zip!.git/m.stl", "referenced"), ("sub/.git/objects/m.stl", "referenced")]
    assert report.left_out == (_audit.GIT_LEFT_OUT,) and "left out: the checkout's own .git" in report.text()


@contextmanager
def nothing_decompressed() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        real = _audit.container_of
        patch.setattr(_audit, "container_of", lambda data: None if real(data) in ("gzip", "bzip2", "xz") else
                      real(data))
        yield


@contextmanager
def no_base64() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_audit, "_base64_blocks", lambda data: iter(()))
        yield


@contextmanager
def every_git_pruned() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        import os

        real = os.walk

        def pruned(top, *args, **keywords):  # type: ignore[no-untyped-def]
            for directory, folders, files in real(top, *args, **keywords):
                folders[:] = [each for each in folders if each != ".git"]
                yield directory, folders, files

        patch.setattr(_audit.os, "walk", pruned)
        yield


@pytest.mark.parametrize("scenario,plant", [
    (lambda tmp: test_a_compressed_copy_is_found_inside_its_stream(None, tmp, "gzip"), nothing_decompressed),
    (lambda tmp: test_a_compressed_copy_is_found_inside_its_stream(None, tmp, "xz"), nothing_decompressed),
    (lambda tmp: test_a_base64_block_in_a_text_file_is_read_at_the_stated_floor_and_over(None, tmp), no_base64),
    (lambda tmp: test_no_directory_is_skipped_by_name_but_a_checkouts_own_git(None, tmp), every_git_pruned)],
    ids=["gzip", "xz", "base64", "git"])
def test_control_each_reach_planted_away_misses_its_copy_every_run(tmp_path, scenario, plant):
    scenario(tmp_path / "unplanted")
    caught = 0
    for run in range(RUNS):
        with plant():
            try:
                scenario(tmp_path / str(run))
            except AssertionError:
                caught += 1
    assert caught == RUNS


# --- the limits refuse, never skip ----------------------------------------------------------------------------------


@pytest.mark.parametrize("limit,value,files", [
    ("MEMBER_LIMIT_BYTES", 4096, {"big.zip": zipped({"a.txt": b"a text it would search, " * 400})}),
    ("MEMBER_LIMIT_BYTES", 4096, {"big.zip": zipped({"a.zip": zipped({"b": bytes(8192)}) + bytes(8192)})}),
    ("MEMBER_LIMIT_BYTES", 4096, {"big.gz": gzip.compress(bytes(8192))}),
    ("MEMBER_LIMIT_BYTES", 2048, {"big.txt": b"a text it would search, " * 200}),
    ("EXPANDED_LIMIT_BYTES", 4096, {"one.zip": zipped({"a": bytes(3000)}), "two.zip": zipped({"b": bytes(3000)})}),
    ("COUNT_LIMIT", 5, {f"f{i}.txt": b"x" for i in range(6)}),
])
def test_each_limit_refuses_rather_than_skips(cache, tmp_path, monkeypatch, limit, value, files):
    from mime.build import BuildRefused

    monkeypatch.setattr(_audit, limit, value)
    with pytest.raises(BuildRefused) as refused:
        assets.audit(tree(tmp_path / "t", files), cache=False)
    assert refused.value.codes() == ["asset.audit"]


def test_a_large_file_of_no_container_kind_is_hashed_as_it_is_read_and_found(cache, tmp_path, monkeypatch):
    monkeypatch.setattr(_audit, "MEMBER_LIMIT_BYTES", 1024)
    big = stl(40, 3)
    root = tree(tmp_path / "t", {"r.json": json.dumps(manifest("big", big, "LicenseRef-Maker")).encode(),
                                 "pkg/big.bin": big, "pkg/loose.bin": stl(41, 4)})
    assert kinds(assets.audit(root, cache=False)) == [("pkg/big.bin", "referenced"),
                                                      ("pkg/loose.bin", "unrecorded mesh")]


def unreadable(name: str) -> bytes:
    """An archive or a stream that cannot be read to its end: a zip member's byte flipped (its CRC fails), a gzip
    stream cut in half, an xz stream's blocks zeroed."""
    if name == "crc.zip":
        held = zipped({"m.stl": CLOSED})
        return held[:60] + bytes([held[60] ^ 0xFF]) + held[61:]
    if name == "cut.gz":
        held = gzip.compress(CLOSED, mtime=0)
        return held[: len(held) // 2]
    return lzma.compress(CLOSED)[:12] + bytes(64)


@pytest.mark.parametrize("name", ["crc.zip", "cut.gz", "bad.xz"])
def test_an_archive_or_a_stream_that_cannot_be_read_is_refused_with_its_code(cache, tmp_path, name):
    from mime.build import BuildRefused

    data = unreadable(name)
    assert _audit.container_of(data) is not None
    with pytest.raises(BuildRefused) as refused:
        assets.audit(tree(tmp_path / "t", {name: data}), cache=False)
    assert refused.value.codes() == ["asset.audit"] and name in str(refused.value)


def test_a_link_in_a_tree_is_not_followed_and_is_counted(cache, tmp_path):
    root = tree(tmp_path / "t", {"a.txt": b"x"})
    (tmp_path / "outside.stl").write_bytes(CLOSED)
    (root / "linked.stl").symlink_to(tmp_path / "outside.stl")
    report = assets.audit(root, cache=False)
    assert report.findings == () and any("1 links, not followed" in each for each in report.left_out)


def test_a_directory_or_a_file_that_cannot_be_read_is_refused_never_skipped(cache, tmp_path):
    import os

    from mime.build import BuildRefused

    for name in ("locked", "locked.stl"):
        root = tree(tmp_path / name, {"a.txt": b"x", "locked/m.stl": CLOSED, "locked.stl": CLOSED})
        held = root / name
        os.chmod(held, 0)
        try:
            if os.access(held, os.R_OK):
                pytest.skip("this user reads what it cannot read (root)")
            with pytest.raises(BuildRefused) as refused:
                assets.audit(root, cache=False)
            assert refused.value.codes() == ["asset.audit"] and "cannot be read" in str(refused.value)
        finally:
            os.chmod(held, 0o700)
