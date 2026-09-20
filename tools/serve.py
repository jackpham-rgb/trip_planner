#!/usr/bin/env python
"""A static file server for local testing that disables caching.

`python -m http.server` works too, but sends no Cache-Control header, so the
browser applies its own heuristic freshness rule -- edits to styles.css/
app.js/recommender.js can silently keep serving a stale cached copy for a
while, which is confusing during active development. This is the same
static site either way; only the headers differ.

    python tools/serve.py [--port 8000]
"""
import argparse
import http.server
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    os.chdir(ROOT)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), NoCacheHandler)
    print(f"Serving {ROOT} at http://127.0.0.1:{args.port}/ (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)
