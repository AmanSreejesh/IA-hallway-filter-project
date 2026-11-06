"""Compatibility entry point for Ishan's updated procedural 3D helmet."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ishan.helmet_updated import HelmetFilter


if __name__ == "__main__":
    from common.main import main
    sys.exit(main(["helmet", *sys.argv[1:]]))
