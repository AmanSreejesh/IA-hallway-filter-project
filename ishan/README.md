# Ishan's Filters

Run `python common/main.py helmet` from the repo root (`astronaut` is an alias).
Both `python ishan/helmet_filter.py` and the original
`python ishan/helmet_Test.py` command launch the same shared system.

The upstream `helmet_filter.py` entry point uses the shared rendering implementation
in `astronaut_filter.py`, `AstronautFilter.apply(frame, face, opacity)`,
using `helmet.png`. It retains the prototype's proportions and face anchor,
rotates with the measured head tilt, and scales with head distance and pose.
The visor remains transparent so the visitor sees their own face.

The shared system owns detection, the 30-inch opt-in zone, immediate activation,
fade from 0.1 opacity, 60-degree angle boundary, multi-user tracking, and
4:15 p.m. cutoff. There are no separate distance or height restrictions here.
See [acceptance procedures](../ACCEPTANCE.md).
