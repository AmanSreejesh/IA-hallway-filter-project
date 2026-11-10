"""
Installs any missing Python packages the first time main.py runs, so a fresh Raspberry Pi only
needs `python3 common/main.py`. Packages that are already there are left alone.

Only uses the standard library, because it runs before OpenCV or NumPy can be imported.
"""
import importlib
import importlib.util
import os
import site
import subprocess
import sys

# (import name, what pip installs, needed to run?)  Keep in sync with requirements.txt.
PACKAGES = [
    ("cv2", "opencv-contrib-python>=4.11,<6", True),
    ("numpy", "numpy>=1.26,<3", True),
    ("mediapipe", "mediapipe>=1.2,<2", False),   # without it: YuNet tracking, Sharingan skipped
    ("moderngl", "moderngl", False),            # without it: 3D emojis drawn with OpenCV
    ("ntplib", "ntplib", False),                # internet time for the schedule (without it: a web page's Date stamp)
    ("tzdata", "tzdata", False),                # timezone list, for computers that don't have one (Windows)
]
# Optional packages that couldn't be installed on this computer, so startup doesn't retry every time.
# Delete this file to try again.
SKIPPED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".install_skipped")


def missing_packages():
    return [(name, spec, needed) for name, spec, needed in PACKAGES if importlib.util.find_spec(name) is None]


def pip_install(spec):
    """Install one package from a prebuilt download (never a long build from source)."""
    command = [sys.executable, "-m", "pip", "install", "--only-binary", ":all:", spec]
    result = subprocess.run(command, capture_output=True, text=True)
    output = result.stdout + result.stderr
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if result.returncode and "externally-managed-environment" in output and not in_venv:
        # Raspberry Pi OS blocks pip for the system Python. This installs into your home folder
        # (~/.local) instead, which doesn't touch the system's own packages.
        result = subprocess.run(command + ["--user", "--break-system-packages"], capture_output=True, text=True)
        output = result.stdout + result.stderr
    return result.returncode == 0, output


def read_skipped():
    try:
        with open(SKIPPED_PATH) as f:
            return {line.strip() for line in f if line.strip()}
    except OSError:
        return set()


def ensure_packages():
    skipped = read_skipped()
    missing = [m for m in missing_packages() if m[2] or m[1] not in skipped]
    if not missing:
        return
    if importlib.util.find_spec("pip") is None:
        names = ", ".join(spec for _, spec, _ in missing)
        print(f"Missing packages ({names}) and pip isn't installed to get them.\n"
              "  Install pip with: sudo apt install python3-pip   then run this again.")
        if any(needed for _, _, needed in missing):
            sys.exit(1)
        return

    for name, spec, needed in missing:
        print(f"Installing {spec} (first run only, this can take a few minutes)...", flush=True)
        ok, output = pip_install(spec)
        if ok:
            continue
        last_line = next((line for line in reversed(output.strip().splitlines()) if line.strip()), "")
        print(f"  Couldn't install {spec}: {last_line}")
        if needed:
            print(f"  main.py can't run without it. Try by hand: {sys.executable} -m pip install \"{spec}\"")
            sys.exit(1)
        print("  Carrying on without it (won't try again; delete common/.install_skipped to retry).")
        try:
            with open(SKIPPED_PATH, "a") as f:
                f.write(spec + "\n")
        except OSError:
            pass

    # A home-folder install made just now isn't on the import path yet
    user_site = site.getusersitepackages()
    if os.path.isdir(user_site) and user_site not in sys.path:
        site.addsitedir(user_site)
    importlib.invalidate_caches()
