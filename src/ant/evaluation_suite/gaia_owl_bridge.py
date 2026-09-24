"""HTTP bridge exposing a `GaiaToolRegistry` instance to the OWL/Workforce
subprocess, which runs under its own isolated `.venv-owl` (camel-ai has
dependency constraints -- notably an `mcp` version pin -- that conflict
with this project's main environment; see docs/owl_gaia.md for why it is
never installed into the shared `.venv`).

WHY A SUBPROCESS + HTTP BRIDGE AT ALL: `owl_gaia.py` (this process) is the
only thing that holds a `GaiaToolRegistry` -- the single object this whole
GAIA substrate funnels every method's tool access through (see gaia_tools
.py's own module docstring on the fairness invariant). OWL's own package
cannot be imported here without the dependency conflict above, so the
Workforce process runs separately (`.venv-owl`'s python, a different
interpreter entirely) and reaches the registry's five primitives only
through this bridge -- never by importing `ant.agents.gaia_tools` itself,
which it structurally cannot do since that package is not on its
sys.path. This keeps OWL's own native toolkits (SearchToolkit,
CodeExecutionToolkit, ...) fully out of reach: the ONLY network/compute
capability the OWL subprocess is ever given is whatever HTTP calls this
bridge serves, and this bridge serves exactly the registry's five
methods, nothing else.

Runs as a background thread inside the calling process (owl_gaia.py),
bound to 127.0.0.1 on an OS-assigned ephemeral port -- one bridge per
question, torn down when that question's OWL subprocess exits, so two
concurrent GAIA questions never share a bridge or a registry.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ant.agents.gaia_tools import GaiaToolRegistry, ToolUnavailableError
from ant.evaluation_suite.gaia_scope import GaiaSubstrateError

_ROUTES = ("search", "open_url", "inspect_file", "inspect_table", "run_python")


def _serialize(tool: str, result: object) -> dict:
    if tool == "search":
        return {"hits": [{"title": h.title, "url": h.url, "snippet": h.snippet} for h in result]}
    if tool == "inspect_table":
        return {"text": result.to_text()}  # type: ignore[union-attr]
    # open_url, inspect_file, run_python all already return plain strings
    return {"text": result}


def _make_handler(registry: GaiaToolRegistry) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib signature
            pass  # silence per-request access logging; gaia_call_log already records everything

        def do_POST(self) -> None:  # noqa: N802 - stdlib method name
            tool = self.path.strip("/")
            if tool not in _ROUTES:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"unknown tool {tool!r}"}).encode())
                return
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            try:
                result = registry.invoke(tool, **body)
                payload = _serialize(tool, result)
                status = 200
            except (ToolUnavailableError, GaiaSubstrateError) as exc:
                payload = {"error": str(exc)}
                status = 200  # a substrate error is a valid tool OUTCOME, not a bridge failure
            except Exception as exc:  # noqa: BLE001 - surfaced to the OWL agent as a tool error
                payload = {"error": repr(exc)}
                status = 200
            body_bytes = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body_bytes)))
            self.end_headers()
            self.wfile.write(body_bytes)

    return Handler


class GaiaOwlBridge:
    """Starts on construction, stops on `close()` / context-manager exit."""

    def __init__(self, registry: GaiaToolRegistry) -> None:
        handler = _make_handler(registry)
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def __enter__(self) -> "GaiaOwlBridge":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
