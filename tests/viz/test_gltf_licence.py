"""The code copied in under MIT, and the inlined viewer's licences: every Python file of ``mime/viz/_gltf/`` carries
exactly one SPDX line, the MIT ones naming usd2gltf 0.3.5 as their origin; the folder's ``LICENSE`` is usd2gltf's,
byte for byte; the viewer's folder holds the licence of every part its bundle's own comments name, and the bundle
names no other licence; NOTICE names both; and the wheel's metadata states the expression with every file (the inlined Mermaid's licences are
``tests/build/test_mermaid_licence.py``'s). The
control: a file planted with a second SPDX line, or with none, is found."""

import hashlib
import re
from pathlib import Path

from mime.viz._web import VIEWER_FILE

ROOT = Path(__file__).resolve().parents[2]
GLTF = ROOT / "src" / "mime" / "viz" / "_gltf"
VIEWER = ROOT / "src" / "mime" / "viz" / "_viewer"
#: The sha256 of usd2gltf 0.3.5's LICENSE, from its PyPI sdist (sha256 6bc4d9db...c05291).
USD2GLTF_LICENSE_SHA256 = "467694d0ab1eac70ae935935982b4f83380ac1a2e1704762e997eff2d602b14c"
ORIGIN = "Derived from usd2gltf 0.3.5 (https://github.com/mikelyndon/usd2gltf), MIT, Copyright (c) 2020 mikelyndon."
#: Each licence text beside the viewer, by its sha256 as its npm package ships it.
VIEWER_LICENCES = {
    "LICENSE.model-viewer": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    "LICENSE.three": "8b378ebe60e2fe500158cb0ac71cb5e8b7d92953c2abcc63a0eb90499653b5bc",
    "LICENSE.lit": "45d31799d0db956cc3eb5469346abbd9b7025babc5ff29fab10d7095da992ef1",
    "LICENSE.fflate": "805f6cb28bb8b6d3a0badd83c93bccd9671fa01a3b4b92b7042b0743325ac243",
    "LICENSE.gainmap-js": "7fb8f722609c6669a9c9a4865e2a86205c7760e7dc7a039b0fb9913c3e62c6cc",
}
SPDX = re.compile(r"^# SPDX-License-Identifier: (.+)$", re.MULTILINE)


def spdx_problems(folder: Path) -> list[str]:
    found: list[str] = []
    for path in sorted(folder.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        lines = SPDX.findall(text)
        if len(lines) != 1:
            found.append(f"{path.name}: {len(lines)} SPDX lines")
        elif lines[0] == "MIT" and ORIGIN not in text:
            found.append(f"{path.name}: MIT with no origin")
        elif lines[0] not in ("MIT", "Apache-2.0"):
            found.append(f"{path.name}: {lines[0]}")
    return found


def test_every_file_of_the_copy_has_one_spdx_line_and_the_mit_ones_name_their_origin():
    assert spdx_problems(GLTF) == []
    mit = sorted(p.name for p in GLTF.glob("*.py") if SPDX.findall(p.read_text(encoding="utf-8")) == ["MIT"])
    assert mit == ["_common.py", "_converter.py", "_mesh.py", "_xform.py"]


def test_control_a_second_spdx_line_or_none_is_found(tmp_path):
    (tmp_path / "a.py").write_text("# SPDX-License-Identifier: MIT\n# SPDX-License-Identifier: Apache-2.0\n")
    (tmp_path / "b.py").write_text("x = 1\n")
    (tmp_path / "c.py").write_text("# SPDX-License-Identifier: MIT\n")
    assert spdx_problems(tmp_path) == ["a.py: 2 SPDX lines", "b.py: 0 SPDX lines", "c.py: MIT with no origin"]


def test_the_copys_licence_is_usd2gltfs_byte_for_byte():
    assert hashlib.sha256((GLTF / "LICENSE").read_bytes()).hexdigest() == USD2GLTF_LICENSE_SHA256
    assert "6bc4d9db4c4960a5ee4e13dfa376585b7ebc9d488f9fd8af79d6fe1d8cc05291" in (GLTF / "README.md").read_text()


def test_the_viewer_carries_the_licence_of_every_part_its_bundle_names():
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in VIEWER.glob("LICENSE.*")} == VIEWER_LICENCES
    named = set(re.findall(r"SPDX-License-Identifier: ([\w.-]+)", (VIEWER / VIEWER_FILE).read_text(encoding="utf-8")))
    assert named == {"BSD-3-Clause", "MIT"}, "a licence the bundle names that no text here covers"
    assert (VIEWER / "LICENSE.model-viewer").read_bytes() == (ROOT / "LICENSE").read_bytes()


def test_notice_names_both_and_the_project_states_the_expression():
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    assert "usd2gltf 0.3.5" in notice and "src/mime/viz/_gltf/" in notice and "<model-viewer> 4.3.1" in notice
    assert all(name in notice for name in ("three.js", "lit", "fflate", "gainmap-js"))
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'license = "Apache-2.0 AND MIT AND BSD-3-Clause AND ISC AND Unlicense"' in pyproject
