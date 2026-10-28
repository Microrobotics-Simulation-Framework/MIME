#!/usr/bin/env python
"""Write ``src/mime/data/env/bases.json`` from Docker Hub: for each Python series MIME supports, the base image's
reference pinned by its digest, with its Python version, its Debian release, its glibc and the Python distributions it
ships (pip), each read from the image itself (its configuration, ``/etc/os-release``, ``/etc/debian_version``, dpkg's
record of ``libc6`` and the ``.dist-info`` folders of its site-packages).

Run it by hand, and land what it writes in a pull request: a base changes when the maintainer chooses, never in a
build. Standard library only; it reads Docker Hub's registry anonymously and downloads the image's layers (tens of
megabytes each, the Debian base layer shared by the series).

    python scripts/bump_bases.py            # writes the file
    python scripts/bump_bases.py --check    # exits 1 when the tags have moved since the file was written
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASES = ROOT / "src" / "mime" / "data" / "env" / "bases.json"
SERIES = ("3.12", "3.13", "3.14")
#: Debian's stable release when the file was first written, for every series alike.
DEBIAN = "trixie"
REPOSITORY = "library/python"
REGISTRY = "https://registry-1.docker.io/v2"
INDEX_TYPES = ("application/vnd.oci.image.index.v1+json",
               "application/vnd.docker.distribution.manifest.list.v2+json")
MANIFEST_TYPES = ("application/vnd.oci.image.manifest.v1+json",
                  "application/vnd.docker.distribution.manifest.v2+json")
WANTED = {"etc/os-release", "usr/lib/os-release", "etc/debian_version", "var/lib/dpkg/status"}


def _token() -> str:
    url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{REPOSITORY}:pull"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)["token"]


def _get(path: str, token: str, accept: tuple[str, ...] = ()) -> tuple[bytes, str]:
    request = urllib.request.Request(f"{REGISTRY}/{REPOSITORY}/{path}",
                                     headers={"Authorization": f"Bearer {token}", "Accept": ", ".join(accept)})
    with urllib.request.urlopen(request, timeout=600) as response:
        return response.read(), response.headers.get("Docker-Content-Digest", "")


def _digest(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


_DIST_INFO = re.compile(r"^usr/local/lib/python3\.\d+/site-packages/([^/]+)-([^/-]+)\.dist-info/METADATA$")


def _files(layer: bytes) -> tuple[dict[str, bytes], dict[str, str]]:
    """The wanted files a layer holds, and the distributions whose metadata it holds, by name."""
    found: dict[str, bytes] = {}
    distributions: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(layer)), mode="r:") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            if name in WANTED and member.isfile():
                held = archive.extractfile(member)
                if held is not None:
                    found[name] = held.read()
            if (match := _DIST_INFO.match(name)) is not None:
                distributions[re.sub(r"[-_.]+", "-", match.group(1)).lower()] = match.group(2)
    return found, distributions


def base(series: str, token: str, layers: dict[str, tuple[dict[str, bytes], dict[str, str]]]) -> dict[str, object]:
    tag = f"{series}-slim-{DEBIAN}"
    index, header = _get(f"manifests/{tag}", token, INDEX_TYPES)
    index_digest = _digest(index)
    if header and header != index_digest:
        raise SystemExit(f"{tag}: the registry names the index {header}, its bytes hash to {index_digest}")
    (amd64,) = [m for m in json.loads(index)["manifests"]
                if m.get("platform", {}).get("os") == "linux" and m["platform"].get("architecture") == "amd64"
                and not m["platform"].get("variant")]
    manifest, _ = _get(f"manifests/{amd64['digest']}", token, MANIFEST_TYPES)
    if _digest(manifest) != amd64["digest"]:
        raise SystemExit(f"{tag}: the linux/amd64 manifest's bytes do not hash to {amd64['digest']}")
    held = json.loads(manifest)
    config_bytes, _ = _get(f"blobs/{held['config']['digest']}", token)
    env = dict(item.split("=", 1) for item in json.loads(config_bytes)["config"]["Env"])
    files: dict[str, bytes] = {}
    distributions: dict[str, str] = {}
    for layer in held["layers"]:
        if layer["digest"] not in layers:
            blob, _ = _get(f"blobs/{layer['digest']}", token)
            if _digest(blob) != layer["digest"]:
                raise SystemExit(f"{tag}: a layer's bytes do not hash to {layer['digest']}")
            layers[layer["digest"]] = _files(blob)
        files.update(layers[layer["digest"]][0])
        distributions.update(layers[layer["digest"]][1])
    release = dict(line.split("=", 1) for line in files["usr/lib/os-release"].decode().splitlines() if "=" in line)
    status = files["var/lib/dpkg/status"].decode()
    libc = re.search(r"^Package: libc6\n(?:[^\n]+\n)*?Version: ([0-9.]+)", status, re.MULTILINE)
    if libc is None:
        raise SystemExit(f"{tag}: dpkg's status records no libc6")
    record = {
        "image": f"python:{tag}@{index_digest}",
        "linux_amd64": amd64["digest"],
        "python": env["PYTHON_VERSION"],
        "debian": release["VERSION_CODENAME"].strip('"'),
        "debian_version": files["etc/debian_version"].decode().strip(),
        "glibc": libc.group(1),
        "distributions": dict(sorted(distributions.items())),
    }
    if not record["python"].startswith(series + ".") or record["debian"] != DEBIAN:
        raise SystemExit(f"{tag}: the image holds Python {record['python']} on {record['debian']}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="compare with the file instead of writing it")
    arguments = parser.parse_args()
    token = _token()
    layers: dict[str, tuple[dict[str, bytes], dict[str, str]]] = {}
    found = {series: base(series, token, layers) for series in SERIES}
    text = json.dumps(found, indent=2, sort_keys=True) + "\n"
    if arguments.check:
        moved = text != BASES.read_text(encoding="utf-8")
        print("the tags have moved since bases.json was written" if moved else "bases.json is current")
        return 1 if moved else 0
    BASES.parent.mkdir(parents=True, exist_ok=True)
    BASES.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
