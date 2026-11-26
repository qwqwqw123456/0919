"""分流标定入口：门控与回放逻辑在 doubao_agent.core.calibration。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doubao_agent.core.calibration import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
