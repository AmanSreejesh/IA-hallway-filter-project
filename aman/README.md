# Aman's Filters

`sharingan_improved.py` adapts Aman's original webcam prototype into a drawing
module for the shared system. It requests MediaPipe, uses the 478-point mesh to
place the effect on both irises, clips it to the eyelids, follows the shared head
rotation, and preserves bright eye highlights. The red iris, pupil, rings and
four tomoe are drawn procedurally; no PNG is required.

From the repository root:

```sh
python -m pip install -r requirements.txt
python common/main.py sharingan
python common/main.py              # rotate through all registered filters
```

`n` switches filters, `d` shows distance/angles/FPS, `c` calibrates looking
forward, and `q` quits. Launching `python aman/sharingan_improved.py` also
delegates to the shared main loop.

The opt-in zone, ultrasonic sensor, multi-user tracking, 60-degree cutoff,
fading, timers, switching and schedule are all implemented in `common/main.py`.
This module only draws; it contains no private eligibility settings or camera.
MediaPipe and its model are required to supply iris/eyelid points. If the shared
tracker falls back to YuNet, the shared switcher skips Sharingan and reports
the missing capability.

See [running the display](../README.md#running-the-display) for installation,
camera calibration, ultrasonic setup and deployment checks.
