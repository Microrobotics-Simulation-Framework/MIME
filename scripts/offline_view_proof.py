"""Show by hand that the page `mime visuals web` writes draws with no network: a manual check, never a CI test (CI has
no WebGL, and `scripts/` is never collected).

    python scripts/offline_view_proof.py <run folder | saved folder | stage> [--chrome google-chrome]

It writes two pages of the target into a temporary folder, the viewer inlined (the default) and the viewer from
jsDelivr (`--cdn`), and opens each in headless Chrome (WebGL through SwiftShader) with every network request sent to a
closed local port and every host name unresolvable, its network log recorded, and the DOM read after twenty seconds of
virtual time. The inlined page passes when `<model-viewer>` reports its load and the animation's duration (the page's
`data-loaded` and `data-duration`, written by its own script) and its log holds no request but Chrome's own (its
update, sign-in and safe-browsing services, by host). The control, which must fail every run: the jsDelivr page under
the same block never loads, and its log holds the viewer's request. Prints a JSON report; exits 0 when both hold."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

#: Hosts of Chrome's own background services, which it calls whatever page it opens.
BROWSER_HOSTS = ("google.com", "googleapis.com", "gvt1.com", "withgoogle.com")
FLAGS = ["--headless=new", "--no-first-run", "--no-default-browser-check", "--use-angle=swiftshader",
         "--enable-unsafe-swiftshader", "--disable-background-networking", "--disable-component-update",
         "--disable-sync", "--disable-default-apps", "--no-pings", "--virtual-time-budget=20000"]
BLOCKED = ["--proxy-server=http://127.0.0.1:9", "--proxy-bypass-list=<-loopback>",
           "--host-resolver-rules=MAP * ~NOTFOUND"]


def opened(chrome: str, page: Path, work: Path, name: str) -> dict[str, object]:
    log = work / f"{name}.netlog.json"
    done = subprocess.run([chrome, *FLAGS, *BLOCKED, f"--user-data-dir={work / name}", f"--log-net-log={log}",
                           "--dump-dom", page.resolve().as_uri()], capture_output=True, text=True, timeout=300)
    body = re.search(r"<body[^>]*>", done.stdout)
    attributes = dict(re.findall(r'data-([a-z]+)="([^"]*)"', body.group(0))) if body else {}
    urls = sorted(set(re.findall(r'"url":"([^"]+)"', log.read_text(encoding="utf-8")))) if log.is_file() else []
    page_urls = [url for url in urls if not any((urlparse(url).hostname or "").endswith(host)
                                                 for host in BROWSER_HOSTS)]
    return {"loaded": attributes.get("loaded") == "1", "duration_s": attributes.get("duration"),
            "error": attributes.get("error"), "requests_by_the_page": page_urls, "browser_requests": len(urls)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", type=Path)
    parser.add_argument("--chrome", default="google-chrome")
    arguments = parser.parse_args()
    from mime.viz import web

    with tempfile.TemporaryDirectory() as held:
        work = Path(held)
        inline, result = web(arguments.target, work / "inline.html")
        cdn, _ = web(arguments.target, work / "cdn.html", cdn=True)
        report = {"target": str(arguments.target), "page_bytes": inline.stat().st_size,
                  "simulated_s": result.simulated_s, "slow": result.slow,
                  "inlined": opened(arguments.chrome, inline, work, "inline"),
                  "control_jsdelivr": opened(arguments.chrome, cdn, work, "cdn")}
    inlined, control = report["inlined"], report["control_jsdelivr"]
    assert isinstance(inlined, dict) and isinstance(control, dict)
    report["offline_shown"] = bool(inlined["loaded"] and inlined["duration_s"] and not inlined["requests_by_the_page"])
    report["control_failed_as_it_must"] = bool(not control["loaded"] and any(
        "cdn.jsdelivr.net" in url for url in control["requests_by_the_page"]))
    print(json.dumps(report, indent=2))
    return 0 if report["offline_shown"] and report["control_failed_as_it_must"] else 1


if __name__ == "__main__":
    sys.exit(main())
