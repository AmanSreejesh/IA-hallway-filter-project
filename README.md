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
- Aman's Sharingan effect runs with `python common/main.py sharingan`. All filters use the same opt-in, distance, multi-user, angle and timing features in the main system.
- Ishan's new procedural 3D helmet (`ishan/helmet_updated.py`) runs with `python common/main.py helmet` (`astronaut` is an alias) and Aryan's clown nose with `python common/main.py clown_nose`. Both are included in the default rotation and dashboard wheel. Their original script names also launch the shared system.
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
calibrate forward. Press `z` at a measured 30 inches to calibrate distance.
`l` explicitly starts/stops a local numeric pose log.

While nobody is using the display it shows a resting screen: the time, "Hi, want to try on a
filter?", and the four filters taking turns in a big frame. Nothing starts until someone taps the
screen (tapping one of the filters starts with that one). Then the privacy intro and filters work as
below. Once nobody has been in view for 5 seconds (`RESTING_AFTER_SECONDS`) the resting screen
comes back. `USE_RESTING_SCREEN = False` in `common/main.py` turns it off. In `--window` mode, click
the window or press space to start. The dashboard font is Schibsted Grotesk (SIL Open Font License,
in `common/dashboard/fonts/`).

All four filters activate on the first eligible detection within the inclusive
30-inch (2.5-foot) opt-in zone, with no waiting period. The first visible opacity
is 0.1 at 30 inches, increasing with proximity to 1 at 25 inches. Head angles up to
60 degrees on each axis are allowed, including a 45-degree tilt; a perpendicular
90-degree profile is rejected. Leaving the zone, losing detection, or exceeding
the angle limit clears activation immediately. Re-entering applies the distance fade immediately.
Head movement alone does not revoke activation. Filters turn off immediately at
4:15 p.m. in the display computer's local timezone, including for existing users,
and become available again at 7:30 a.m. Set the computer's timezone and clock
for the school location.

Removing the waiting period supersedes the earlier requirement that visits under
three seconds never show a filter. Brief visitors can receive an effect while
they meet the distance, angle and schedule rules.

There is no floor-height requirement, minimum/maximum face-width gate, or
requirement for the whole face box to fit on screen. A child's face, a seated
visitor, and a close-up follow the same rules while their nose is in frame.
Every filter uses current-frame landmarks, scale and pose to follow movement.
Eight faces are supported by default; use
`--max-faces N` to change capacity and check performance on the display hardware.
Use `--camera N` to choose a camera and `--focal-px N` to calibrate the camera's
distance estimate. At a measured frontal distance, calculate
`new_focal = old_focal * actual_inches / displayed_inches` from debug view.

Distance comes from the camera. The shared tracker measures each person's eye
gap using their iris size, and pressing `z` at 30 inches calibrates the camera.
No ultrasonic sensor is required. Sharingan requires MediaPipe's iris landmarks and is skipped with a
message if only YuNet is available; compatible filters continue running.

The new helmet uses procedural 3D geometry instead of `helmet.png`, with a
see-through visor so the visitor's face remains visible. It uses the same
current-frame pose and distance as the other filters, with no extra camera or
segmentation pass. See [Ishan's filter notes](ishan/README.md).

Tracking and scale continue updating outside the zone, but effects remain hidden
at 5 feet. This resolves the conflict between the 5-foot geometry tests and the
30-inch opt-in requirement. The clown nose retains Aryan's proportional sizing
and placement, with a procedural red nose when the prototype's `clown_nose.png`
is absent. An optional transparent PNG at `aryan/clown_nose.png` replaces it.

Run the deterministic acceptance replays with:

```sh
python -m unittest discover -s tests -v
```

See [ACCEPTANCE.md](ACCEPTANCE.md) for thresholds and live-camera procedures.
Automated tests use synthetic detections and real renderers; they do not establish
physical pass rates. Distance relies on camera calibration and measured eye
spacing, particularly for children. Validate the camera calibration,
lighting, seated and child heights, and maximum group size at the actual display
before recording the required 10/10 or 5/5 results.

## Labels to create
- user: on purpose, user: passerby, user: maintainer, user: excluded
- priority: high, priority: medium, priority: low
