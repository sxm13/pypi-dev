import os
import sys

pkg_dir = os.path.dirname(__file__)
if pkg_dir not in sys.path:
    sys.path.append(pkg_dir)

from .tobacco import run_tobacco