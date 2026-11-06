# Ishan's Filters

Run `python common/main.py helmet` from the repo root (`astronaut` is an alias).
`python ishan/helmet_updated.py`, `python ishan/helmet_filter.py`, and the original
`python ishan/helmet_Test.py` all launch the same shared system.

`helmet_updated.py` implements `HelmetFilter.apply(frame, face, opacity)` using
Ishan's new procedural 3D dome, side hardware, collar, curved visor and rails.
It uses each person's current shared nose position, face size and all three head
angles, with no additional pose smoothing. Geometry is built once and drawing is
clipped to the visible camera patch, including during close-ups. The visor has a
slight tint and reflections while preserving the visitor's actual face details.
All artwork follows the opacity supplied by the shared system.

The uploaded standalone camera, YuNet download and segmentation pass have been
replaced by the shared camera/tracker. This filter requires no separate PNG or
segmentation model. `helmet_filter.py` and `astronaut_filter.py` remain compatible
entry points for the updated renderer; `helmet.png` is retained as the older asset.

The shared system owns detection, the 30-inch opt-in zone, immediate activation,
distance fade from 0.1 opacity at 30 inches to full opacity at 25 inches,
60-degree angle boundary, multi-user tracking, and the 4:15 p.m. to 7:30 a.m.
schedule. There are no separate distance or height restrictions here.
See [acceptance procedures](../ACCEPTANCE.md).
