#!/usr/bin/env python
"""Run the web app (FastAPI backend + the static PWA frontend in web/).

    python scripts/run_web.py [--port 8420] [--reload]
"""
import argparse
import os
import sys
from pathlib import Path

SRC = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, SRC)
# --reload spawns a fresh subprocess that doesn't inherit sys.path.insert above,
# only the environment -- so PYTHONPATH has to be set explicitly for it too.
os.environ["PYTHONPATH"] = SRC + os.pathsep + os.environ.get("PYTHONPATH", "")

import uvicorn  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run("recsys.api:app", host="127.0.0.1", port=args.port, reload=args.reload,
                reload_dirs=[SRC, str(Path(__file__).parent.parent / "web")])
