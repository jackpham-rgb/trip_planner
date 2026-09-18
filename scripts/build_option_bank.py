#!/usr/bin/env python
"""Thin CLI wrapper: rebuild data/option_bank.json from the master workbook.

    python scripts/build_option_bank.py [--xlsx PATH] [--out PATH]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recsys.data_loader import main  # noqa: E402

if __name__ == "__main__":
    main()
