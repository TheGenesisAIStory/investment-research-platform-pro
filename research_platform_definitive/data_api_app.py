from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
APP_DIR = PROJECT_ROOT / "research_platform_app"
for candidate in [PROJECT_ROOT, PROJECT_ROOT / "src", APP_DIR]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from research_platform_app.data_api_view import render_data_api_control_center


render_data_api_control_center()
