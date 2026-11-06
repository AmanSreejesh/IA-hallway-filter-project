# Shared filter acceptance

All rules apply to Emoji, Sharingan, Helmet (Astronaut), and Clown Nose through
`common/main.py`. Filters only draw; the old standalone filenames delegate to
that entry point. Choose a filter on the dashboard wheel to hold it during a
trial, or run `python common/main.py FILTER --window` for one filter.
Use `helmet` or its `astronaut` alias for Ishan's filter; both resolve to the
same wheel entry and renderer.
The helmet uses Ishan's new procedural 3D mesh in `ishan/helmet_updated.py`,
including a see-through visor so the visitor's face remains visible. It consumes
the same current-frame per-person pose, scale and opacity as the other filters.

Effects show at 30 inches or closer and are hidden beyond it. Opacity is set by
distance: 0.1 at exactly 30 inches, rising linearly to fully visible at 25 inches
and closer. Every head angle up to and including 60 degrees counts; past 60 the
effect stops immediately. A new visitor must stay eligible (in the zone, facing the
screen) for 3 seconds before their effect starts; meanwhile the dashboard shows the
privacy note in the middle of the screen with a 3-2-1 countdown. A blink or missed
frame under half a second doesn't restart it, and someone who steps away and comes
back within 5 seconds doesn't see it again. People hurrying past in under 3 seconds
never get an effect (#7).

Timer (#8): a 1-minute timer starts when someone's effect starts. When it's up, a
small note gently says 1 minute has passed (for 10 seconds); at 3 minutes a more
prominent banner reminds them to get back to what they were doing. Neither stops or
covers the effect. The camera losing their face for a moment doesn't
restart it; stepping away or looking away for more than 5 seconds does.

| Scenario, repeated for each filter | Expected behavior | Required physical result |
| --- | --- | --- |
| Child approaches within 30 inches | After the 3-second privacy intro, effect appears, correctly placed for the child's face height | 10/10 |
| Visitor enters within 30 inches with a 45-degree head tilt | After the 3-second privacy intro, effect appears and tilts in the same direction; repeat both tilt directions | 10/10 |
| Visitor sits in a chair within 30 inches | After the 3-second privacy intro, effect appears at seated face height | 10/10 |
| Stand between 25 and 30 inches, looking at the screen | See-through effect (0.1 at 30 inches up to full at 25) | 100% |
| Several people at 2.5 feet, each tilting their head | Effect on everyone, turning with each head; stops past 60 degrees | 5/5 |
| Approach from 2.5 feet to 0.5 feet | Scale tracks continuously; at/inside 30 inches, after the 3-second intro, effect covers its intended feature, including the whole nose for Clown Nose | At least 4/5 |
| At 2 feet, sway 30 degrees left and right | Effect stays visible and attached throughout the sway | 5/5, attachment maintained throughout each trial |
| Walk perpendicular to the camera along a path whose closest point is 5 feet away, crossing in 2 seconds | No effect at any point | 5/5 |
| Walk in from 30 inches at a 60-degree turn | Opacity 0.1 at 30 inches, increasing as they come closer, full at 25 inches | 5/5 |
| Hurry through view in less than 3 seconds, including inside the zone | No effect at any point, even while facing forward | At least 9/10 |
| Sit in the zone with an effect for 1 minute | Note that 1 minute has passed | 5/5 |
| Stay with an effect for 3 minutes | Prominent reminder to get back to what they were doing | 100% |
| Enter view at or after 4:15 p.m. local time; also test someone already active at cutoff | No effect from the cutoff until 7:30 a.m., including during a fade; effects work again from 7:30 a.m. | 100% (record trial count) |

## Automated checks

Run from the repository root with the project environment:

```sh
python -m unittest discover -s tests -v
```

`tests/test_acceptance.py` loads all four registered filters and exercises their
actual renderers with synthetic landmark/pose streams. It checks repeated child,
seated and tilted placements, immediate first opacity, angles, brief eligible visits,
schedule boundaries, independent person timers, the distance fade, exact distances
for faces off-centre, turned and tilted, the 30-inch calibration for any eye gap,
loss/re-entry, close-up clipping, nose coverage and scale, and helmet rotation
and see-through visor, yaw/pitch/roll tracking without pose history, and opacity
applied to the entire 3D helmet. Tests do not open the webcam, download models or save
camera frames. Direct rendering outside the zone is only a geometry check;
normal processing is separately asserted to keep the effect hidden there.

## Physical trial setup and record

Use the deployed camera, display, lighting and installation timezone. Mark
30 inches, 5 feet and 0.5 feet from the camera/filter installation. Ensure the
camera can see both standing children's faces and seated faces. Enable debug
view (`d`) to check distance and angles; calibrate forward (`c`) while looking
at the display. Then set the zone: stand with your eyes exactly 30 inches from
the camera (tape measure), look at the screen and press `z` (or "Set 30-inch
distance" in the staff controls). This measures the camera's focal length, so
the debug view reads exactly 30 inches there; it is saved and loaded on every
start. Distance comes from eye spacing, and each person's own eye gap is
measured against their iris (about 11.7 mm wide in everyone), so children and
adults with closer- or wider-set eyes read correctly too. The debug view shows
"eye gap ... (measured)" once a person has been measured, usually within a
second while they face the screen. Verify the boundary with each participant.

Repeat the table for each filter and record filter, participant/height condition,
distance source, trial number, observed behavior, and pass/fail. For approaches,
check that activation begins on entry to the zone without a pause, and check
the final close-up. Measure sway continuously, including at the endpoints.
For the schedule test, use the installation's local clock; do not disable the
schedule to get a daytime acceptance result after dismissal.

Physical results: **not yet measured**. Synthetic replays verify software
behavior given detections; they do not establish face-detection reliability,
camera calibration accuracy, frame rate or the requested physical success rates.
