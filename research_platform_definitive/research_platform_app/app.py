from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in [APP_DIR, PROJECT_ROOT, SRC_ROOT]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from home_command_center import render_home_command_center
from support import configure_page


configure_page("Command Center")
render_home_command_center()
