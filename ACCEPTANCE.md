# Shared filter acceptance

All rules apply to Emoji, Sharingan, Helmet (Astronaut), and Clown Nose through
`common/main.py`. Filters only draw; the old standalone filenames delegate to
that entry point. Choose a filter on the dashboard wheel to hold it during a
trial, or run `python common/main.py FILTER --window` for one filter.
Use `helmet` or its `astronaut` alias for Ishan's filter; both resolve to the
same wheel entry and renderer.

The agreed interpretation keeps effects hidden beyond 30 inches. Five-foot
approach/sway tests check tracked geometry there, with visible artwork checked
after entry. The three-second continuous eligibility requirement applies to
every new activation, including children, seated visitors and angled visitors.
A visit shorter than three seconds must never produce even a faint effect.

| Scenario, repeated for each filter | Expected behavior | Required physical result |
| --- | --- | --- |
| Child approaches within 30 inches, stays facing display for at least 3 seconds | Effect appears, correctly placed for the child's face height | 10/10 |
| Visitor enters within 30 inches with a 45-degree head tilt and stays 3 seconds | Effect appears and tilts in the same direction; repeat both tilt directions | 10/10 |
| Visitor sits in a chair within 30 inches and stays 3 seconds | Effect appears at seated face height | 10/10 |
| Approach from 5 feet to 0.5 feet | Scale tracks continuously; no effect beyond 30 inches or before dwell; at/inside 30 inches after dwell, effect covers its intended feature, including the whole nose for Clown Nose | At least 4/5 |
| At 5 feet, sway 30 degrees left and right | Tracking remains attached; effect stays hidden. Automated direct-render tests inspect nose attachment outside the zone. Also repeat inside the zone after dwell to observe visible attachment throughout the sway | 5/5, attachment maintained throughout each trial |
| Walk perpendicular to the camera along a path whose closest point is 5 feet away, crossing in 2 seconds | No effect at any point | 5/5 |
| Enter within 30 inches at a 60-degree turn and remain eligible 3 seconds | First visible opacity is 0.1, increasing while detection continues, reaching full opacity after 0.4 seconds | 5/5 |
| Hurry through view in less than 3 seconds, including inside the zone | No effect at any point, even while facing forward | At least 9/10 |
| Enter view at or after 4:15 p.m. local time; also test someone already active at cutoff | No effect on or after the cutoff, including during a fade | 100% (record trial count) |

## Automated checks

Run from the repository root with the project environment:

```sh
python -m unittest discover -s tests -v
```

`tests/test_acceptance.py` loads all four registered filters and exercises their
actual renderers with synthetic landmark/pose streams. It checks repeated child,
seated and tilted placements, exact first opacity, angles, short crossings,
schedule boundaries, independent person timers, sensor-based zone decisions,
loss/re-entry, close-up clipping, nose coverage and scale, and helmet rotation
and transparent visor. Tests do not open the webcam, download models or save
camera frames. Direct rendering outside the zone is only a geometry check;
normal processing is separately asserted to keep the effect hidden there.

## Physical trial setup and record

Use the deployed camera, display, lighting and installation timezone. Mark
30 inches, 5 feet and 0.5 feet from the camera/filter installation. Ensure the
camera can see both standing children's faces and seated faces. Enable debug
view (`d`) to check distance and angles; calibrate forward (`c`) while looking
at the display. At a measured frontal distance calculate
`new_focal = old_focal * actual_inches / displayed_inches` and restart with
`--focal-px new_focal`. Verify the boundary with each participant: camera-only
distance uses average eye spacing and cannot guarantee absolute inches across
all face sizes. Use the calibrated ultrasonic setup when available and repeat
the trials with its camera fallback. One ultrasonic sensor anchors the nearest
face, rather than measuring every visitor independently.

Repeat the table for each filter and record filter, participant/height condition,
distance source, trial number, observed behavior, and pass/fail. For approaches,
pause inside the zone for three seconds before the final close-up so activation
is observable. An instantaneous approach entirely under three seconds should
remain hidden instead. Measure sway continuously, including at the endpoints.
For the schedule test, use the installation's local clock; do not disable the
schedule to get a daytime acceptance result after dismissal.

Physical results: **not yet measured**. Synthetic replays verify software
behavior given detections; they do not establish face-detection reliability,
camera calibration accuracy, frame rate or the requested physical success rates.
