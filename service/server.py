"""A small JSON API around the trained rider policy (standard library only).

    uv run python serve.py --port 8000

    GET  /health   -> {"status": "ok", "policy": {...}}
    GET  /schema   -> the fields a state may contain and the 16 actions
    POST /act      -> body {"state": {...}} or {"observation": [94 numbers]}
                      returns the recommended action, the legal actions and the policy's scores
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from environment.custom_env import ACTION_NAMES, OBS_DIM
from service.agent import STATE_FIELDS, RiderAgent


def make_handler(agent: RiderAgent):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send(200, {"status": "ok", "policy": agent.meta})
            elif self.path == "/schema":
                self._send(200, {"state_fields": STATE_FIELDS, "observation_size": OBS_DIM, "actions": list(ACTION_NAMES)})
            else:
                self._send(404, {"error": "use /health, /schema or POST /act"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/act":
                self._send(404, {"error": "use POST /act"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                result = agent.recommend(state=body.get("state"), observation=body.get("observation"))
            except (ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
                self._send(400, {"error": str(error)})
                return
            self._send(200, result)

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            print("api:", format % args, flush=True)

    return Handler


def serve(model: str = "models/best_agent.pt", host: str = "127.0.0.1", port: int = 8000) -> None:
    agent = RiderAgent.load(model)
    server = ThreadingHTTPServer((host, port), make_handler(agent))
    print(f"FARUMASI rider policy ({agent.meta['algo']} experiment {agent.meta['experiment']}) on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the trained rider policy as a JSON API")
    parser.add_argument("--model", default="models/best_agent.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(args.model, args.host, args.port)


if __name__ == "__main__":
    main()
