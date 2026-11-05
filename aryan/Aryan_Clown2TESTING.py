"""Compatibility launcher: Aryan's filter now uses the shared hallway system."""
from pathlib import Path
import sys


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.main import main
    sys.exit(main(["clown_nose", *sys.argv[1:]]))
