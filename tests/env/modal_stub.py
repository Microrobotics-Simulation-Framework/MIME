"""A stand-in for ``modal`` that records what a generated script asks of it, so that the script's image, mounts and
functions are read by importing the script, as ``modal run`` does, with no Modal account, call or network.

``load(script, tmp)`` writes the stand-in into ``tmp``, imports the script in a fresh interpreter that finds it first,
and returns what the script defined: the image's steps in order, the functions with their settings, the local
entrypoints, the volumes, and which top-level modules the import brought in. A stub ``uv`` on the path writes a fixed
wheel for the dev path, whose script builds the checkout's wheel."""

import json
import os
import subprocess
import sys
from pathlib import Path

STUB = '''
__version__ = "0.0-stub"
RECORD = {"images": [], "functions": {}, "entrypoints": [], "volumes": {}}


def is_local():
    return True


class Image:
    def __init__(self, steps):
        self.steps = steps

    @staticmethod
    def from_registry(tag, **options):
        return Image([{"step": "from_registry", "tag": tag, **options}])

    def _then(self, step):
        image = Image([*self.steps, step])
        RECORD["images"].append(image.steps)
        return image

    def uv_pip_install(self, *packages, **options):
        return self._then({"step": "uv_pip_install", "packages": list(packages), **options})

    def add_local_file(self, local_path, remote_path, *, copy=False):
        return self._then({"step": "add_local_file", "local": str(local_path), "remote": remote_path, "copy": copy})

    def add_local_dir(self, local_path, remote_path, *, copy=False, **options):
        return self._then({"step": "add_local_dir", "local": str(local_path), "remote": remote_path, "copy": copy})

    @property
    def object_id(self):
        import os

        held = os.environ.get("STUB_IMAGE_ID")
        if held is None:
            raise RuntimeError("not hydrated")
        return held


class Volume:
    def __init__(self, name, read_only=False):
        self.name, self.read_only = name, read_only

    @staticmethod
    def from_name(name, **options):
        RECORD["volumes"][name] = options
        return Volume(name)

    def with_mount_options(self, read_only=False):
        return Volume(self.name, read_only)

    def commit(self):
        pass


class App:
    def __init__(self, name):
        RECORD["app"] = name

    def function(self, **settings):
        def wrap(function):
            held = {key: value for key, value in settings.items() if key not in ("image", "volumes")}
            held["image"] = settings["image"].steps
            held["volumes"] = {path: [volume.name, volume.read_only] for path, volume in settings["volumes"].items()}
            RECORD["functions"][function.__name__] = held
            return function
        return wrap

    def local_entrypoint(self):
        def wrap(function):
            RECORD["entrypoints"].append(function.__name__)
            return function
        return wrap
'''

LOADER = '''
import importlib.util, json, sys
before = set(sys.modules)
spec = importlib.util.spec_from_file_location("modal_app", sys.argv[1])
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)
import modal
record = dict(modal.RECORD)
record["image"] = script.image.steps
record["runner"], record["info"] = script._runner()
record["imported"] = sorted({name.split(".")[0] for name in set(sys.modules) - before})
calls = []
script._mime = lambda *arguments: calls.append(list(arguments))
script._check("/experiment", "/out/env-check.json")
record["check_arguments"] = calls[0]
record["builder"] = __import__("os").environ.get("MODAL_IMAGE_BUILDER_VERSION")
print(json.dumps(record))
'''

UV = '''#!{python}
import pathlib, sys
arguments = sys.argv[1:]
out = pathlib.Path(arguments[arguments.index("--out-dir") + 1])
out.mkdir(parents=True, exist_ok=True)
(out / "mime_engine-0.0.0-py3-none-any.whl").write_bytes(b"a wheel the stub built")
'''


def load(script: Path, tmp: Path) -> dict:
    stub = tmp / "stub"
    (stub / "modal").mkdir(parents=True, exist_ok=True)
    (stub / "modal" / "__init__.py").write_text(STUB, encoding="utf-8")
    uv = stub / "bin" / "uv"
    uv.parent.mkdir(exist_ok=True)
    uv.write_text(UV.format(python=sys.executable), encoding="utf-8")
    uv.chmod(0o755)
    environment = {**os.environ, "PYTHONPATH": str(stub), "PATH": f"{uv.parent}{os.pathsep}{os.environ['PATH']}"}
    environment.pop("MODAL_IMAGE_BUILDER_VERSION", None)
    done = subprocess.run([sys.executable, "-c", LOADER, str(script)], capture_output=True, text=True,
                          env=environment, cwd=tmp, timeout=120, check=False)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.splitlines()[-1])
