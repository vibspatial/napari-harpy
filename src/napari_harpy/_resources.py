from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
LOGO_PATH = PACKAGE_ROOT / "_static" / "logo.png"


def get_logo_path() -> Path:
    return LOGO_PATH
