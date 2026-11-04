# IA Hallway Filter Project
Team: Viraj Das, Aman Sreejesh, Ishan Prabhu, Aryan Mali

## How this repo works
- **Issues** hold our user stories. Use the "User story" template.
- **constraints.md** holds the facts every story must respect.
- **test-cases.md** holds tests that trace back to stories by issue number.
- **prototype/** holds snapshots of our code when it passes a story's tests.

## Where code goes
- **viraj/, aman/, ishan/, aryan/** each hold one team member's filter(s). Work only in your own folder.
- **common/main.py** is the main system. It combines everyone's filters and implements the shared features (opt-in zone, fade-in, timer, multi-user, engagement).
- Run everything with `python common/main.py`, or only your filter with `python common/main.py emoji`. See the top of `common/main.py` for how to plug in a filter.
- Aman's Sharingan effect runs with `python common/main.py sharingan`. All filters use the same opt-in, ultrasonic, multi-user, angle and timing features in the main system.
- Install with `python -m pip install -r requirements.txt`.

## Running the display

Use Python 3.11 in a desktop session with camera access:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python common/main.py
```

The first run downloads the face model into `common/`; later runs reuse it.
Model files and machine-specific calibration are excluded from Git. Camera
frames are never saved. Press `q` or Escape to quit, `n` to switch filters,
`d` for distance/angle/FPS diagnostics, and `c` while looking at the screen to
calibrate forward. `l` explicitly starts/stops a local numeric pose log.

All filters share the 30-inch (2.5-foot) opt-in zone, 0.5-second dwell,
0.4-second fade, independent per-person rotation, and immediate cutoff at
45 degrees on any head axis. Eight faces are supported by default; use
`--max-faces N` to change capacity and check performance on the display hardware.
Use `--camera N` to choose a camera and `--focal-px N` to calibrate the camera's
distance estimate. At a measured frontal distance, calculate
`new_focal = old_focal * actual_inches / displayed_inches` from debug view.

On Raspberry Pi, use a virtual environment created with `--system-site-packages`
to access the OS-provided `gpiozero`. Follow the HC-SR04 wiring and voltage
divider instructions in `common/main.py`. The shared ultrasonic reading anchors
the nearest face's camera distance and scales the other estimates; one sensor
cannot measure every person independently. Invalid readings fall back to the
camera. Sharingan requires MediaPipe's iris landmarks and is skipped with a
message if only YuNet is available; compatible filters continue running.

Before deployment, verify both filters with the actual camera, lighting and
maximum group size at 30 inches, including opposite head tilts and the 45-degree
cutoff. Repeat five times with the ultrasonic sensor enabled and with camera
estimates. Physical camera/sensor acceptance trials remain unverified.

## Labels to create
- user: on purpose, user: passerby, user: maintainer, user: excluded
- priority: high, priority: medium, priority: low
