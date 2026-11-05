# Aryan's Filters

Run `python common/main.py clown_nose` from the repo root. The original
`python aryan/Aryan_Clown2TESTING.py` command launches the same shared system.

`clown_nose_filter.py` implements `ClownNoseFilter.apply(frame, face, opacity)`.
It keeps the prototype's relative sizing and upward offset, rotates with the
head, and uses current nose coordinates without smoothing lag. The shared
system controls the camera, 30-inch zone, 3-second dwell, angle limit, fade,
multi-user tracking, and 4:15 p.m. cutoff for every filter.

The uploaded prototype referenced a missing `clown_nose.png`. A procedural red
rubber nose is included; place a transparent BGRA PNG with that name in this
folder to use custom artwork. No separate model or camera loop is needed.

See [acceptance procedures](../ACCEPTANCE.md). Tracking at five feet is tested
without making the effect visible outside the 30-inch zone.
