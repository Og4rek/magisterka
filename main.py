"""Compatibility entry point for the PCam reproduction CLI."""

from __future__ import annotations

import sys
from pathlib import Path


SOURCE_DIRECTORY = Path(__file__).resolve().parent / "src"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from pcam_repro.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
