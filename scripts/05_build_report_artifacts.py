#!/usr/bin/env python3
"""Build plots and other report artifacts from existing outputs/ (no model run)."""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.reporting.run import main

if __name__ == "__main__":
    raise SystemExit(main(project_root=project_root))
