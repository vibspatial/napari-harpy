from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
# Single source of truth for the logo; the README links to this file too.
LOGO_PATH = PACKAGE_ROOT / "_static" / "logo.svg"


def get_logo_path() -> Path:
    return LOGO_PATH
