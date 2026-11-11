"""Deterministic acceptance replays, without a camera, model download or GPIO.

These verify shared decisions and artwork against synthetic measured geometry.
Live detector accuracy and physical pass rates require the trials in ACCEPTANCE.md.
"""
import contextlib
from datetime import datetime, timezone
import io
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from common import main as system
from common.overlay import draw_artwork, premultiply
from aman.sharingan_improved import EYES
from aryan.clown_nose_filter import ClownNoseFilter
from ishan.astronaut_filter import AstronautFilter
from ishan.helmet_updated import HelmetFilter, head_rotation
from viraj.emoji_filter import rotation


def detection(distance=24, center=(320, 240), roll=0, yaw=0, pitch=0, eye_gap=None, gaze=0.0):
    """Generate current-frame points, including iris mesh, for someone `distance` inches away
    (straight-line, like a tape measure) with the point between their eyes at `center`.
    eye_gap is the person's real eye gap in inches (default: the average); their iris is the
    usual IRIS_MM wide. gaze shifts both irises sideways, as a fraction of the eye's width."""
    eye_gap = eye_gap or system.EYE_GAP_INCHES
    ray = math.hypot((center[0] - 320) / system.FOCAL_PX, (center[1] - 240) / system.FOCAL_PX, 1.0)
    size = system.FOCAL_PX * eye_gap * ray / distance
    iris_r = system.IRIS_MM / 25.4 / eye_gap / 2   # iris radius, in eye gaps
    angle = math.radians(roll)
    rotate = np.array([[math.cos(angle), -math.sin(angle)],
                       [math.sin(angle), math.cos(angle)]])
    turn = math.cos(math.radians(yaw))
    eyes_at = np.array([0, -0.55]) @ rotate.T * size   # so the eyes' midpoint lands on `center`

    def pixels(points):
        return (np.asarray(points) * [turn, 1]) @ rotate.T * size + center - eyes_at

    points = pixels([[-0.5, -0.55], [0.5, -0.55], [0, 0], [-0.4, 0.6], [0.4, 0.6]])
    row = np.array([center[0]-1.3*size, center[1]-1.5*size, 2.6*size, 3*size,
                    *points.ravel(), 1.0], dtype=np.float32)
    mesh = np.tile(np.asarray(center, dtype=np.float32), (478, 1))
    for eye_x, (contour, iris, ring) in zip((-0.5, 0.5), EYES):
        t = np.arange(16) * 2 * math.pi / 16
        mesh[list(contour)] = pixels(np.column_stack((eye_x-0.24*np.cos(t), -0.55+0.12*np.sin(t))))
        look = eye_x + gaze * 0.48
        mesh[iris] = pixels([look, -0.55])
        mesh[list(ring)] = pixels([[look+iris_r, -0.55], [look, -0.55-iris_r],
                                  [look-iris_r, -0.55], [look, -0.55+iris_r]])
    mesh[1] = points[2]
    return row, roll, (yaw, pitch), mesh


class RecordingFilter:
    def __init__(self, actual):
        self.actual = actual
        self.calls = []
        self.forgotten = []

    def apply(self, frame, face, opacity):
        self.calls.append((face.id, opacity))
        self.actual.apply(frame, face, opacity)

    def forget(self, face_id):
        self.forgotten.append(face_id)


class AcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Exercise the real registry and real renderers, including Emoji's CPU fallback.
        with patch("viraj.emoji_filter.RENDERER", "software"), contextlib.redirect_stdout(io.StringIO()):
            cls.filters = system.load_filters([])

    def setUp(self):
        self.forward = patch.dict(system.FORWARD, yaw=0.0, pitch=0.0)
        self.forward.start()
        self.addCleanup(self.forward.stop)
        # Most tests check what a filter does once it's on, so they skip the 3-second privacy intro.
        # The DwellTests below check the intro itself.
        self.dwell = patch.object(system, "DWELL_SECONDS", 0.0)
        self.dwell.start()
        self.addCleanup(self.dwell.stop)

    def step(self, faces, effect, now, detections=(), *, off=False):
        next_id = [max(faces, default=-1)+1]
        system.track(faces, detections, now, next_id, (640, 480))
        frame = np.full((480, 640, 3), 140, np.uint8)
        result = system.process_faces(frame, faces, [effect], effect, now, 0.1,
                                      filters_off=off, draw_notes=False)
        return frame, result

    def test_registry_loads_every_filter(self):
        self.assertEqual([f.name for f in self.filters], ["Emoji", "Sharingan", "Helmet", "Clown Nose"])
        self.assertEqual(len(self.filters), len(system.FILTERS))

    def test_upstream_helmet_command_and_astronaut_alias_share_one_filter(self):
        for names in (["helmet"], ["astronaut"], ["helmet", "astronaut"]):
            with self.subTest(names=names), contextlib.redirect_stdout(io.StringIO()):
                filters = system.load_filters(names)
                self.assertEqual(len(filters), 1)
                self.assertEqual(filters[0].name, "Helmet")
                self.assertIs(type(filters[0]), HelmetFilter)

    def test_updated_helmet_is_used_by_registry_and_legacy_entry_points(self):
        from ishan.helmet_filter import HelmetFilter as LegacyHelmetFilter
        self.assertEqual(system.FILTERS["helmet"], ("ishan.helmet_updated", "HelmetFilter"))
        self.assertIs(LegacyHelmetFilter, HelmetFilter)
        self.assertIsInstance(AstronautFilter(), HelmetFilter)

    def test_short_child_seated_and_45_degree_tilt_ten_trials_per_filter(self):
        for actual in self.filters:
            for trial in range(10):
                # Height affects image placement, never eligibility. Test both roll directions.
                for center, roll in [((260, 300), 0), ((320, 240), (-1)**trial*45), ((420, 350), 0)]:
                    with self.subTest(filter=actual.name, trial=trial, center=center, roll=roll):
                        faces, effect = {}, RecordingFilter(actual)
                        frame, _ = self.step(faces, effect, 10, [detection(30, center, roll)])
                        self.assertEqual(len(effect.calls), 1)
                        self.assertAlmostEqual(effect.calls[0][1], 0.1, places=4)   # edge of the zone
                        self.assertAlmostEqual(faces[0].roll, roll)
                        self.assertFalse(np.all(frame == 140), "actual renderer must draw")

    def test_eligible_visits_activate_without_waiting(self):
        for actual in self.filters:
            with self.subTest(filter=actual.name):
                faces, effect = {}, RecordingFilter(actual)
                # With the privacy intro off, a brief, fast crossing activates while eligible.
                for elapsed in (0, 0.1, 0.2):
                    center = (160+1000*elapsed, 240)
                    frame, _ = self.step(faces, effect, 10+elapsed, [detection(24, center)])
                    self.assertFalse(np.all(frame == 140))
                    self.assertEqual(len(effect.calls), round(elapsed*10)+1)
                self.assertAlmostEqual(effect.calls[0][1], 1.0)   # 24 in: inside 25, fully visible

    def test_45_degree_turn_starts_at_point_one_and_increases_five_trials(self):
        # Up to 45 degrees the fade is by distance: 0.1 at the 30-inch edge, rising as they come
        # closer, full at 25.
        for actual in self.filters:
            for trial in range(5):
                with self.subTest(filter=actual.name, trial=trial):
                    faces, effect = {}, RecordingFilter(actual)
                    for now, inches in ((10.0, 30), (10.1, 28.5), (10.2, 27), (10.3, 25)):
                        self.step(faces, effect, now, [detection(inches, yaw=(-1)**trial*45)])
                    opacity = [value for _, value in effect.calls]
                    self.assertAlmostEqual(opacity[0], 0.1, places=4)
                    self.assertTrue(all(a < b for a, b in zip(opacity, opacity[1:])))
                    self.assertAlmostEqual(opacity[-1], 1, places=4)

    def test_turned_past_45_appears_at_point_one_and_rises_with_time_five_trials(self):
        # Issue #5: turned past 45 degrees (up to and including 90), the filter only appears at 0.1
        # and increases with time in detection, reaching full after FADE_IN_SECONDS.
        for yaw in (46, 60, -60, 75, 90, -90):
            for trial in range(5):
                with self.subTest(yaw=yaw, trial=trial):
                    faces, effect = {}, RecordingFilter(self.filters[trial % len(self.filters)])
                    ramp = system.FADE_IN_SECONDS
                    for k in (0, 0.25, 0.5, 0.75, 1.0, 1.5):
                        row, _, _, mesh = detection(24)
                        self.step(faces, effect, 10 + k * ramp, [(row, 0, (yaw, 0), mesh)])
                    opacity = [value for _, value in effect.calls]
                    self.assertEqual(len(opacity), 6)
                    self.assertAlmostEqual(opacity[0], 0.1, places=6)
                    self.assertAlmostEqual(opacity[1], 0.1 + 0.9 * 0.25, places=6)
                    self.assertTrue(all(a < b for a, b in zip(opacity[:5], opacity[1:5])))
                    self.assertAlmostEqual(opacity[4], 1, places=6)

    def test_up_to_45_degrees_uses_the_distance_fade_only(self):
        for yaw in (0, 30, 45, -45):
            with self.subTest(yaw=yaw):
                faces, effect = {}, RecordingFilter(self.filters[0])
                self.step(faces, effect, 10, [detection(24, yaw=yaw)])
                self.assertAlmostEqual(effect.calls[0][1], 1)   # 24 in: full straight away

    def test_turning_past_45_keeps_a_filter_that_is_already_up(self):
        faces, effect = {}, RecordingFilter(self.filters[0])
        for now in (10.0, 10.5, 11.0, 11.5):
            self.step(faces, effect, now, [detection(24)])
        row, _, _, mesh = detection(24)
        self.step(faces, effect, 11.6, [(row, 0, (80, 0), mesh)])
        self.assertAlmostEqual(effect.calls[-1][1], 1)
        self.assertAlmostEqual(faces[0].distance, 24, places=2)   # distance kept while turned

    def turned(self, yaw, inches=24):
        row, _, _, mesh = detection(inches)
        return (row, 0, (yaw, 0), mesh)

    def test_entering_turned_then_facing_the_screen_switches_to_distance(self):
        faces, effect = {}, RecordingFilter(self.filters[0])
        self.step(faces, effect, 10.0, [self.turned(70)])
        self.step(faces, effect, 10.4, [self.turned(70)])
        self.assertAlmostEqual(effect.calls[-1][1], 0.1 + 0.9 * 0.4 / system.FADE_IN_SECONDS, places=6)
        self.step(faces, effect, 10.45, [self.turned(20)])
        self.assertAlmostEqual(effect.calls[-1][1], 1)    # faced the screen: distance (24 in = full)
        self.step(faces, effect, 10.5, [self.turned(70)])
        self.assertAlmostEqual(effect.calls[-1][1], 1)    # turning away again stays distance-based

    def test_entering_facing_the_screen_never_gets_the_time_fade(self):
        faces, effect = {}, RecordingFilter(self.filters[0])
        for now in (10.0, 10.5):
            self.step(faces, effect, now, [detection(24)])
        self.step(faces, effect, 10.6, [self.turned(95)])   # past 90: filter hidden for a moment
        self.assertEqual(faces[0].opacity, 0)
        self.step(faces, effect, 10.7, [self.turned(70)])   # back within 90, still turned
        self.assertAlmostEqual(effect.calls[-1][1], 1)      # straight back at full, no fade-in

    def test_fade_is_linear_by_distance_between_25_and_30_inches(self):
        for inches, expected in [(6, 1), (24.9, 1), (25, 1), (26, 0.82), (27.5, 0.55), (29, 0.28),
                                 (30, 0.1), (30.01, 0), (40, 0)]:
            with self.subTest(inches=inches):
                self.assertAlmostEqual(system.zone_opacity(inches), expected, places=6)

    def test_distance_is_exact_off_centre_turned_and_tilted(self):
        # A real pinhole projection: eyes 2.5 in apart anywhere in front of the camera, turned up
        # to 60 degrees and tilted up to 45, as long as the camera can see the face (60 degrees).
        f, size = system.FOCAL_PX, (640, 480)
        for x, y, z in [(0, 0, 30), (-10, 6, 26), (9, -7, 24), (12, 8, 20), (-6, -8, 28)]:
            for yaw in (-60, -30, 0, 30, 60):
                for roll in (-45, 0, 45):
                    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
                    eye = np.array([c*math.cos(math.radians(roll)), c*math.sin(math.radians(roll)), s])
                    nose = np.array([s, 0, -c])
                    middle = np.array([x, y, z], float)
                    if np.degrees(np.arccos(nose @ -middle / np.linalg.norm(middle))) > 60:
                        continue   # seen from the side: the camera can't see both eyes
                    pixel = lambda p: (f*p[0]/p[2] + size[0]/2, f*p[1]/p[2] + size[1]/2)
                    points = np.zeros((7, 2))
                    points[2], points[3] = pixel(middle - eye*1.25), pixel(middle + eye*1.25)
                    points[4] = pixel(middle + nose)
                    with self.subTest(position=(x, y, z), yaw=yaw, roll=roll):
                        true = float(np.linalg.norm(middle))
                        self.assertAlmostEqual(system.estimate_distance(points, yaw, size), true,
                                               delta=true*0.002)

    def settle(self, eye_gap, inches=30, frames=12, center=(330, 230), start=0.0, face=None):
        """Track one still person for a few frames so their iris measurement settles."""
        face = face or system.Face(0)
        for k in range(frames):
            row, roll, pose, mesh = detection(inches, center, eye_gap=eye_gap)
            system.update_face(face, row, roll, pose, start + k / 30, (640, 480), mesh)
        return face

    def test_children_and_adults_both_read_thirty_at_thirty_inches(self):
        # Eye gaps from young children to wide-set adults. The iris is the same size in all of them.
        for gap in (1.9, 2.1, 2.3, 2.5, 2.7, 2.9):
            with self.subTest(eye_gap_mm=round(gap * 25.4)), patch.object(system, "DISTANCE_FOCAL", None):
                face = self.settle(gap, frames=system.IRIS_MIN_SAMPLES - 1)
                averaged = face.distance        # before enough frames: the average eye gap is assumed
                face = self.settle(gap, frames=2, start=0.3, face=face)
                self.assertAlmostEqual(face.eye_gap, gap, places=3)
                self.assertAlmostEqual(face.distance, 30, places=2)
                self.assertAlmostEqual(averaged, 30 * system.EYE_GAP_INCHES / gap, places=2)

    def test_walking_in_does_not_change_the_measured_eye_gap(self):
        # The iris and the eye gap grow together as someone walks closer, so the ratio holds.
        with patch.object(system, "DISTANCE_FOCAL", None):
            face = system.Face(0)
            for k, inches in enumerate(np.linspace(48, 22, 30)):
                row, roll, pose, mesh = detection(float(inches), eye_gap=2.1)
                system.update_face(face, row, roll, pose, k / 30, (640, 480), mesh)
                face.speed = 0.0   # the walk-in itself, not blur, is what this checks
            self.assertAlmostEqual(face.eye_gap, 2.1, places=3)
            self.assertAlmostEqual(face.distance, 22, places=2)

    def test_looking_sideways_turning_or_blur_are_not_measured(self):
        for kwargs, speed in [({"gaze": 0.5}, 0), ({"gaze": -0.5}, 0), ({"yaw": 30}, 0), ({}, 3.0)]:
            with self.subTest(kwargs=kwargs, speed=speed):
                face = system.Face(0)
                for k in range(20):
                    row, roll, pose, mesh = detection(30, eye_gap=2.0, **kwargs)
                    system.update_face(face, row, roll, pose, k / 30, (640, 480), mesh)
                    face.speed = speed
                    face.iris_samples.clear() if speed and k == 0 else None
                    if speed:
                        self.assertIsNone(system.iris_sample(face, (640, 480)))
                if not speed:
                    self.assertFalse(face.iris_samples)
                    self.assertIsNone(face.eye_gap)

    def test_pressing_z_measures_the_camera_whoever_presses_it(self):
        size = (640, 480)
        for presser in (2.0, 2.5, 2.8):   # a child, an average adult, a wide-set adult
            with self.subTest(presser=presser), patch.object(system, "DISTANCE_FOCAL", 450.0):  # wrong guess
                person = self.settle(presser, center=(380, 200))
                focal = system.calibration_sample(person, size) / person.eye_gap
                self.assertAlmostEqual(focal, system.FOCAL_PX, places=2)
                with patch.object(system, "DISTANCE_FOCAL", focal):
                    for other in (1.9, 2.5, 2.9):
                        for inches in (25, 29.99, 30, 30.01):
                            face = self.settle(other, inches, center=(300, 260))
                            self.assertAlmostEqual(face.distance, inches, places=2)

    def test_profile_five_feet_two_second_crossing_five_trials(self):
        for actual in self.filters:
            for trial in range(5):
                faces, effect = {}, RecordingFilter(actual)
                for elapsed in np.linspace(0, 2, 21):
                    self.step(faces, effect, 10+elapsed, [detection(60, (80+240*elapsed, 240), yaw=90)])
                self.assertFalse(effect.calls, actual.name)

    def test_in_zone_profile_and_invalid_pose_never_draw_over_time(self):
        for actual in self.filters:
            for pose in [(90.01, 0, 0), (0, 90, 0), (0, 0, 90), (float("nan"), 0, 0), (0, 60.01, 0), (0, 0, 60.01)]:
                with self.subTest(filter=actual.name, pose=pose):
                    faces, effect = {}, RecordingFilter(actual)
                    yaw, pitch, roll = pose
                    # Corrupt pose is injected after building otherwise valid landmark geometry.
                    row, _, _, mesh = detection()
                    for now in np.arange(10, 15, 0.1):
                        self.step(faces, effect, float(now), [(row, roll, (yaw, pitch), mesh)])
                    self.assertFalse(effect.calls)

    def test_zone_boundary_and_immediate_revocation(self):
        for actual in self.filters:
            faces, effect = {}, RecordingFilter(actual)
            for now in np.arange(10, 13.5, 0.1):
                self.step(faces, effect, float(now), [detection(30)])
            self.assertTrue(effect.calls)
            for now, det in [(13.5, detection(30.01)), (13.6, detection(yaw=90.01))]:
                calls = len(effect.calls)
                self.step(faces, effect, now, [det])
                self.assertEqual(len(effect.calls), calls)
                self.assertEqual(faces[0].opacity, 0)

    def test_schedule_exact_cutoff_new_and_existing_visitors(self):
        for stamp, expected in [("16:14:59.999999", False), ("16:15:00", True),
                                ("16:15:00.000001", True), ("23:59:59", True), ("00:00:00", True),
                                ("07:29:59.999999", True), ("07:30:00", False), ("12:00:00", False)]:
            self.assertEqual(system.filters_are_off(datetime.fromisoformat("2026-11-05T"+stamp)), expected)
        for actual in self.filters:
            faces, effect = {}, RecordingFilter(actual)
            for now in np.arange(10, 13.5, 0.1):
                self.step(faces, effect, float(now), [detection()])
            self.assertTrue(effect.calls)
            calls = len(effect.calls)
            for now in np.arange(13.5, 18, 0.1):
                self.step(faces, effect, float(now), [detection()], off=True)
                self.assertEqual(faces[0].opacity, 0)
            self.assertEqual(len(effect.calls), calls)
            faces, effect = {}, RecordingFilter(actual)
            for now in np.arange(20, 24, 0.1):
                self.step(faces, effect, float(now), [detection()], off=True)
            self.assertFalse(effect.calls)

    def test_reentry_restarts_fade_immediately_and_cleanup(self):
        for actual in self.filters:
            faces, effect = {}, RecordingFilter(actual)
            self.step(faces, effect, 10, [detection()])
            self.step(faces, effect, 10.4, [detection()])
            self.assertAlmostEqual(faces[0].opacity, 1)
            calls = len(effect.calls)
            self.step(faces, effect, 10.5)  # missed detection clears the effect
            self.assertEqual(faces[0].opacity, 0)
            self.assertEqual(len(effect.calls), calls)
            self.step(faces, effect, 10.6, [detection()])
            self.assertAlmostEqual(effect.calls[-1][1], 1.0)   # back immediately, at 24 in
            self.assertEqual(len(effect.calls), calls+1)
            self.step(faces, effect, 11.2)
            self.assertIn(0, faces)   # their timer is still running, so they're kept for a moment
            self.step(faces, effect, 10.6 + system.TIMER_RESET_SECONDS + 0.1)
            self.assertEqual(faces, {})
            self.assertEqual(effect.forgotten, [0])

    def run_timer(self, until, gaps=(), start=100.0, step=0.5):
        """One person in the zone from `start` to `until` seconds later, with the camera missing
        them during each (from, to) gap. Returns the (gentle, firm) seen at every step."""
        faces, effect = {}, RecordingFilter(self.filters[0])
        seen = []
        for t in np.arange(0, until + 1e-9, step):
            missing = any(a <= t < b for a, b in gaps)
            _, (_, firm, _, gentle) = self.step(faces, effect, start + float(t), [] if missing else [detection()])
            seen.append((float(t), gentle, firm))
        return faces, seen

    def test_one_minute_note_shows_after_a_minute(self):
        _, seen = self.run_timer(75)
        gentle_at = [t for t, g, _ in seen if g]
        self.assertTrue(gentle_at)
        self.assertAlmostEqual(gentle_at[0], system.TIMER_GENTLE_SECONDS, delta=0.5)
        self.assertLessEqual(gentle_at[-1], system.TIMER_GENTLE_SECONDS + system.GENTLE_SHOW_SECONDS)
        self.assertFalse(any(f for _, _, f in seen))

    def test_three_minute_reminder_shows_and_stays(self):
        _, seen = self.run_timer(200)
        firm_at = [t for t, _, f in seen if f]
        self.assertAlmostEqual(firm_at[0], system.TIMER_FIRM_SECONDS, delta=0.5)
        self.assertTrue(all(f for t, _, f in seen if t >= system.TIMER_FIRM_SECONDS))

    def test_camera_losing_the_face_briefly_keeps_the_timer(self):
        # Dropouts of 1-4 seconds (detection misses, a glance down at a test) don't restart it
        _, seen = self.run_timer(75, gaps=[(20, 21), (35, 39), (50, 51.5)])
        gentle_at = [t for t, g, _ in seen if g]
        self.assertTrue(gentle_at)
        self.assertAlmostEqual(gentle_at[0], system.TIMER_GENTLE_SECONDS, delta=0.5)

    def test_stepping_away_resets_the_timer(self):
        _, seen = self.run_timer(75, gaps=[(30, 30 + system.TIMER_RESET_SECONDS + 1)])
        self.assertFalse(any(g for t, g, _ in seen if t < 30 + system.TIMER_RESET_SECONDS + 1 + 59))

    def test_no_fade_activates_immediately_at_full_opacity(self):
        with patch.object(system, "USE_FADE_IN", False):
            face = system.Face(0)
            face.distance = 29.5   # in the fading band, but fading is off
            system.update_opacity(face, True, 10, 0.001)
            self.assertEqual(face.opacity, 1)

    def test_independent_multi_user_immediate_activation_and_fade(self):
        for actual in self.filters:
            faces, effect = {}, RecordingFilter(actual)
            first, second = detection(28, (180, 260)), detection(24, (460, 340), roll=-45)
            for now in (10, 10.2, 10.4):
                self.step(faces, effect, now, [first])
            self.assertEqual({fid for fid, _ in effect.calls}, {0})
            self.step(faces, effect, 10.5, [first, second])
            self.assertEqual({fid for fid, _ in effect.calls}, {0, 1})
            self.assertAlmostEqual(faces[0].opacity, 0.46, places=4)   # 28 in: partway faded
            self.assertAlmostEqual(faces[1].opacity, 1, places=4)      # 24 in: fully visible

    def test_tracking_and_resizing_five_feet_to_six_inches_five_trials(self):
        nose = ClownNoseFilter()
        for trial in range(5):
            faces, effect = {}, RecordingFilter(nose)
            widths = []
            for index, distance in enumerate((60, 48, 36, 30, 24, 18, 12, 6)):
                now = 10 + index*0.1
                self.step(faces, effect, now, [detection(distance)])
                self.assertEqual(len(effect.calls), max(0, index-2))
                self.assertEqual(len(faces), 1)
                face = faces[0]
                self.assertAlmostEqual(face.distance, distance, places=4)
                # Draw directly to inspect geometry outside the zone; runtime never draws there.
                frame = np.zeros((480, 640, 3), np.uint8)
                nose.apply(frame, face, 1)
                ys, xs = np.where(frame[:, :, 2] > 30)
                widths.append(xs.max()-xs.min()+1)
                # Cover a nose envelope of half an eye-gap, including its tip.
                nx, ny = np.rint(face.nose).astype(int)
                half = max(1, int(face.size*0.25))
                self.assertTrue(np.all(frame[ny, nx-half:nx+half+1, 2] > 30))
            self.assertTrue(all(a < b for a, b in zip(widths, widths[1:])))
            self.assertEqual(effect.calls[0], (0, 0.1), "activate on entry at 30 inches")

    def test_sway_at_five_feet_has_no_anchor_lag_and_remains_hidden_five_trials(self):
        for actual in self.filters:
            for trial in range(5):
                faces, effect = {}, RecordingFilter(actual)
                # Include both roll and yaw interpretations of a 30-degree sway.
                for index, angle in enumerate((0, -30, 30, -30, 30, 0)):
                    det = detection(60, (320+angle, 240), roll=angle, yaw=angle)
                    self.step(faces, effect, 10+index*0.1, [det])
                    np.testing.assert_allclose(faces[0].nose, det[0][8:10], atol=1e-6)
                    # detection() turns faces with a flat (not perspective) squash, so a face that is
                    # both turned and off-centre isn't exactly 60 in to the real-camera formula.
                    # test_distance_is_exact_off_centre_turned_and_tilted covers exactness.
                    self.assertAlmostEqual(faces[0].distance, 60, delta=60 * 0.03)
                    # Direct rendering checks the attachment while normal processing is hidden.
                    frame = np.zeros((480, 640, 3), np.uint8)
                    ClownNoseFilter().apply(frame, faces[0], 1)
                    nx, ny = np.rint(faces[0].nose).astype(int)
                    self.assertGreater(frame[ny, nx, 2], 30)
                self.assertFalse(effect.calls)

    def test_active_sway_stays_visible_for_every_frame_five_trials(self):
        for actual in self.filters:
            for trial in range(5):
                faces, effect = {}, RecordingFilter(actual)
                for now in np.linspace(10, 13.5, 36):
                    self.step(faces, effect, float(now), [detection()])
                self.assertTrue(effect.calls)
                for index, angle in enumerate((0, -30, 30, -30, 30, 0)):
                    calls = len(effect.calls)
                    self.step(faces, effect, 13.6+index*0.1,
                              [detection(24, (320+angle, 240), roll=angle, yaw=angle)])
                    self.assertEqual(len(faces), 1)
                    self.assertEqual(len(effect.calls), calls+1)
                    self.assertEqual(faces[0].opacity, 1)

    def test_schedule_is_enforced_by_default_processing(self):
        effect = RecordingFilter(ClownNoseFilter())
        faces = {}
        for now in np.linspace(10, 13.1, 32):
            self.step(faces, effect, float(now), [detection()])
        calls = len(effect.calls)
        self.assertGreater(calls, 0)
        with patch.object(system.school_clock, "now", return_value=datetime(2026, 11, 5, 16, 15)):
            faces[0].last_seen = 13.2
            system.process_faces(np.zeros((480, 640, 3), np.uint8), faces, [effect], effect,
                                 13.2, 0.1, draw_notes=False)
        self.assertEqual(len(effect.calls), calls)
        self.assertEqual(faces[0].opacity, 0)

    def test_mediapipe_decode_preserves_inclusive_angle_boundaries(self):
        for roll, yaw, pitch in [(45, 0, 0), (-45, 0, 0), (0, 60, 0), (0, -60, 0), (0, 0, 60)]:
            with self.subTest(roll=roll, yaw=yaw, pitch=pitch):
                _, _, _, mesh = detection(24, roll=roll, yaw=yaw, pitch=pitch)
                matrix = np.eye(4)
                axes = system.MediaPipeDetector.TO_OPENCV
                matrix[:3, :3] = axes @ rotation(roll, yaw, pitch) @ axes
                result = SimpleNamespace(
                    face_landmarks=[[SimpleNamespace(x=x/640, y=y/480) for x, y in mesh]],
                    facial_transformation_matrixes=[matrix])
                dets = system.MediaPipeDetector.decode(result, 640, 480)
                faces = {}
                system.track(faces, dets, 10, [0], (640, 480))
                np.testing.assert_allclose(faces[0].measured_angles, (yaw, pitch, roll), atol=1e-5)
                self.assertTrue(system.within_angle_limit(faces[0]))

    def test_close_up_partial_face_is_eligible_and_artwork_clips(self):
        for actual in self.filters:
            faces, effect = {}, RecordingFilter(actual)
            for now in np.linspace(10, 13.1, 32):
                frame, _ = self.step(faces, effect, float(now), [detection(6, (320, 270))])  # chin off screen
            self.assertTrue(effect.calls)
            self.assertFalse(np.all(frame == 140))

    def test_helmet_visibly_rotates_and_preserves_face_window(self):
        helmet = HelmetFilter()
        original = np.zeros((480, 640, 3), np.uint8)
        tilted = original.copy()
        face = system.Face(0)
        row, tilt, pose, mesh = detection(24)
        system.update_face(face, row, tilt, pose, 10, (640, 480), mesh)
        helmet.apply(original, face, 1)
        # Hold the anchor fixed: isolate the rotation from landmark translation.
        face.roll = 45
        helmet.apply(tilted, face, 1)
        self.assertGreater(np.count_nonzero(original != tilted), 1000)
        nx, ny = np.rint(face.nose).astype(int)
        self.assertTrue(np.all(original[ny, nx] < 15), "visor has only a slight tint")
        # The off-center bottom of the helmet identifies the direction of rotation.
        mask = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) > 20
        rotated_mask = cv2.cvtColor(tilted, cv2.COLOR_BGR2GRAY) > 20
        y, x = np.where(mask)
        ry, rx = np.where(rotated_mask)
        anchor = np.asarray(face.nose)
        expected = (np.array([x.mean(), y.mean()])-anchor) @ np.array([[2**-0.5, 2**-0.5],
                                                                 [-2**-0.5, 2**-0.5]]) + anchor
        np.testing.assert_allclose([rx.mean(), ry.mean()], expected, atol=2)

    def test_3d_helmet_follows_shared_yaw_pitch_and_roll_without_history(self):
        helmet = HelmetFilter()
        face = system.Face(0)
        row, tilt, pose, mesh = detection(24)
        system.update_face(face, row, tilt, pose, 10, (640, 480), mesh)
        base = np.full((480, 640, 3), 140, np.uint8)
        neutral = base.copy()
        helmet.apply(neutral, face, 1)
        for roll, yaw, pitch in [(45, 0, 0), (0, 60, 0), (0, -60, 0), (0, 0, 45)]:
            with self.subTest(roll=roll, yaw=yaw, pitch=pitch):
                face.roll, face.yaw, face.pitch = roll, yaw, pitch
                np.testing.assert_allclose(head_rotation(roll, yaw, pitch), rotation(roll, yaw, pitch))
                turned = base.copy()
                helmet.apply(turned, face, 1)
                self.assertGreater(np.count_nonzero(neutral != turned), 1000)
        # A second person's rendering must not affect a returning person's pose.
        face.id = 1
        helmet.apply(base.copy(), face, 1)
        face.id = 0
        face.roll = face.yaw = face.pitch = 0
        restored = base.copy()
        helmet.apply(restored, face, 1)
        np.testing.assert_array_equal(restored, neutral)

    def test_3d_helmet_preserves_face_detail_and_applies_opacity_to_all_artwork(self):
        helmet = HelmetFilter()
        face = system.Face(0)
        row, tilt, pose, mesh = detection(24)
        system.update_face(face, row, tilt, pose, 10, (640, 480), mesh)
        y, x = np.indices((480, 640))
        pattern = np.where((x//4 + y//4) % 2, 220, 50).astype(np.uint8)
        base = np.repeat(pattern[:, :, None], 3, axis=2)
        full, half, hidden = base.copy(), base.copy(), base.copy()
        helmet.apply(full, face, 1)
        helmet.apply(half, face, 0.5)
        helmet.apply(hidden, face, 0)
        np.testing.assert_array_equal(hidden, base)
        np.testing.assert_allclose(half.astype(float), (base.astype(float)+full)/2, atol=1)
        nx, ny = np.rint(face.nose).astype(int)
        detail = full[ny-10:ny+10, nx-10:nx+10, 0]
        self.assertGreater(np.ptp(detail.astype(int)), 140, "actual face contrast remains visible")
        np.testing.assert_array_equal(full[:30, :30], base[:30, :30])

    def test_overlay_zero_opacity_and_offscreen_are_noops(self):
        art = premultiply(np.full((20, 20, 4), 255, np.uint8))
        for center, opacity in [((320, 240), 0), ((-1000, -1000), 1)]:
            frame = np.full((480, 640, 3), 100, np.uint8)
            draw_artwork(frame, art, center, 100, 45, opacity)
            self.assertTrue(np.all(frame == 100))


if __name__ == "__main__":
    unittest.main()


class DwellTests(unittest.TestCase):
    """Issue #7 and the privacy intro: 3 seconds in the zone, facing the screen, before a filter."""

    @classmethod
    def setUpClass(cls):
        with patch("viraj.emoji_filter.RENDERER", "software"), contextlib.redirect_stdout(io.StringIO()):
            cls.filters = system.load_filters([])

    def setUp(self):
        self.forward = patch.dict(system.FORWARD, yaw=0.0, pitch=0.0)
        self.forward.start()
        self.addCleanup(self.forward.stop)
        self.intro = patch.object(system, "USE_WALK_BY_CHECK", True)
        self.intro.start()
        self.addCleanup(self.intro.stop)

    step = AcceptanceTests.step

    def visit(self, frames):
        """frames: list of (time, detection or None). Returns the times a filter was drawn."""
        faces, effect = {}, RecordingFilter(self.filters[0])
        drawn = []
        for now, det in frames:
            before = len(effect.calls)
            self.step(faces, effect, now, [det] if det is not None else [])
            if len(effect.calls) > before:
                drawn.append(now)
        return faces, drawn

    def test_hurrying_through_in_under_three_seconds_never_gets_a_filter_ten_trials(self):
        for trial in range(10):
            with self.subTest(trial=trial):
                _, drawn = self.visit([(10 + t, detection(26, (150 + 150 * t, 240)))
                                       for t in np.arange(0, 2.9, 0.1)])
                self.assertEqual(drawn, [])

    def test_filter_starts_after_three_seconds_in_the_zone(self):
        faces, drawn = self.visit([(10 + t, detection()) for t in np.arange(0, 4.01, 0.1)])
        self.assertTrue(drawn)
        self.assertAlmostEqual(drawn[0], 10 + system.DWELL_SECONDS, delta=0.1)
        self.assertTrue(faces[0].opted_in)

    def test_a_blink_does_not_restart_the_three_seconds(self):
        frames = [(10 + t, None if 1.5 <= t < 1.8 else detection()) for t in np.arange(0, 4.01, 0.1)]
        _, drawn = self.visit(frames)
        self.assertAlmostEqual(drawn[0], 10 + system.DWELL_SECONDS, delta=0.1)

    def test_stepping_out_restarts_the_three_seconds(self):
        frames = [(10 + t, detection(40) if 1.5 <= t < 2.5 else detection()) for t in np.arange(0, 6.01, 0.1)]
        _, drawn = self.visit(frames)
        self.assertAlmostEqual(drawn[0], 10 + 2.5 + system.DWELL_SECONDS, delta=0.15)

    def test_coming_back_soon_after_skips_the_intro(self):
        frames = [(10 + t, detection()) for t in np.arange(0, 4.01, 0.1)]
        frames += [(14 + t, detection(40)) for t in np.arange(0.1, 2.01, 0.1)]   # steps back 2 s
        frames += [(16 + t, detection()) for t in np.arange(0.1, 1.01, 0.1)]     # comes back
        _, drawn = self.visit(frames)
        self.assertIn(round(16.1, 6), [round(t, 6) for t in drawn])   # straight back on, no 3 s wait


class SchoolClockTests(unittest.TestCase):
    """The schedule and the resting screen's clock use the real time in the school's timezone,
    never the Pi's own clock (which can be wrong). The internet is only needed once, at setup."""

    def setUp(self):
        import tempfile
        clock = system.school_clock
        saved = (clock._base, clock._source, clock._warned, clock._loaded)
        self.addCleanup(lambda: [setattr(clock, k, v) for k, v in
                                 zip(("_base", "_source", "_warned", "_loaded"), saved)])
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        for name, value in (("SAVED_PATH", folder.name + "/clock.json"), ("BOOT_ID_PATH", folder.name + "/boot_id")):
            p = patch.object(clock, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.boot_file = folder.name + "/boot_id"
        self.set_boot("first-power-on")
        clock._base, clock._source, clock._loaded = None, None, True
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    def set_boot(self, boot_id):
        with open(self.boot_file, "w") as f:
            f.write(boot_id)

    def test_keeps_the_setup_time_offline_while_the_pi_stays_on(self):
        clock = system.school_clock
        real = datetime(2026, 11, 5, 21, 10, tzinfo=timezone.utc).timestamp()   # 4:10 pm in New York
        with patch.object(clock, "_from_ntp", return_value=(real, 1000.0)), \
                patch.object(clock.time, "monotonic", return_value=1000.0), \
                patch.object(clock.time, "time", return_value=real + 5 * 86400):   # the Pi's clock is days out
            self.assertTrue(clock.sync())                                       # setup, with internet
        # main.py is closed and started again later, with no internet: same power-on
        clock._base, clock._source, clock._loaded = None, None, False
        with patch.object(clock, "_from_ntp", side_effect=AssertionError("used the internet")), \
                patch.object(clock, "_from_web", side_effect=AssertionError("used the internet")), \
                patch.object(clock.time, "monotonic", return_value=1000.0 + 300), \
                patch.object(clock.time, "time", return_value=0.0):             # Pi clock now nonsense
            t = clock.now()
            self.assertEqual((t.hour, t.minute), (16, 15))
            self.assertTrue(system.filters_are_off())
        self.assertIn("saved", clock.source())

    def test_after_a_restart_carries_the_saved_correction_over(self):
        clock = system.school_clock
        real = datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc).timestamp()    # 7:00 am in New York
        pi_wrong_by = -3 * 3600 - 17                                             # Pi clock 3 h 17 s slow
        with patch.object(clock, "_from_ntp", return_value=(real, 500.0)), \
                patch.object(clock.time, "monotonic", return_value=500.0), \
                patch.object(clock.time, "time", return_value=real + pi_wrong_by):
            clock.sync()
        self.set_boot("second-power-on")                                         # turned off and on again
        clock._base, clock._source, clock._loaded = None, None, False
        later_pi = real + pi_wrong_by + 45 * 60                                  # 45 minutes later, by the Pi
        with patch.object(clock.time, "monotonic", return_value=20.0), patch.object(clock.time, "time", return_value=later_pi):
            t = clock.now()
            self.assertEqual((t.hour, t.minute), (7, 45))
            self.assertFalse(system.filters_are_off())
        self.assertIn("restart", clock.source())

    def test_never_set_up_falls_back_to_the_pi_clock_without_waiting_for_the_internet(self):
        clock = system.school_clock
        clock._loaded = False
        pi = datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc).timestamp()
        with patch.object(clock.time, "time", return_value=pi):
            self.assertEqual((clock.now().hour, str(clock.now().tzinfo)), (7, "America/New_York"))
        self.assertIn("this computer", clock.source())

    def test_ignores_a_wrong_pi_clock(self):
        clock = system.school_clock
        real = datetime(2026, 11, 5, 21, 15, tzinfo=timezone.utc).timestamp()   # 4:15 pm in New York
        with patch.object(clock, "_from_ntp", return_value=(real, 100.0)), \
                patch.object(clock.time, "monotonic", return_value=100.0), \
                patch.object(clock.time, "time", return_value=real - 9 * 3600):   # the Pi is 9 hours out
            self.assertTrue(clock.sync())
            t = clock.now()
            self.assertEqual((t.hour, t.minute, str(t.tzinfo)), (16, 15, "America/New_York"))
            self.assertTrue(system.filters_are_off())   # the cutoff uses the real time
        with patch.object(clock.time, "monotonic", return_value=100.0 - 60):     # a minute earlier
            self.assertFalse(system.filters_are_off())

    def test_keeps_counting_from_the_answer_and_handles_daylight_saving(self):
        clock = system.school_clock
        july = datetime(2026, 7, 1, 11, 29, 30, tzinfo=timezone.utc).timestamp()   # 7:29:30 am EDT
        with patch.object(clock, "_from_ntp", return_value=(july, 50.0)), patch.object(clock.time, "monotonic", return_value=50.0):
            clock.sync()
            self.assertEqual(clock.now().utcoffset().total_seconds(), -4 * 3600)
            self.assertTrue(system.filters_are_off())
        with patch.object(clock.time, "monotonic", return_value=50.0 + 30):     # 30 s later: 7:30
            self.assertFalse(system.filters_are_off())

    def test_falls_back_to_a_web_date_stamp_when_time_servers_are_blocked(self):
        clock = system.school_clock
        real = datetime(2026, 11, 5, 15, 0, tzinfo=timezone.utc).timestamp()
        with patch.object(clock, "_from_ntp", side_effect=OSError("blocked")), \
                patch.object(clock, "_from_web", return_value=(real, 10.0)) as web, \
                patch.object(clock.time, "monotonic", return_value=10.0):
            self.assertTrue(clock.sync())
            self.assertEqual(clock.now().hour, 10)
        web.assert_called()
        self.assertIn("web", clock._source)

    def test_no_internet_is_not_an_error(self):
        clock = system.school_clock
        with patch.object(clock, "_from_ntp", side_effect=OSError), patch.object(clock, "_from_web", side_effect=OSError):
            self.assertFalse(clock.sync())
        pi = datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc).timestamp()
        with patch.object(clock.time, "time", return_value=pi):
            self.assertEqual((clock.now().hour, str(clock.now().tzinfo)), (7, "America/New_York"))

    def test_main_never_reads_the_computer_clock_for_the_time_of_day(self):
        with open(system.__file__) as source:
            code = source.read()
        for banned in ("datetime.now(", "time.time(", "time.localtime(", "date.today("):
            self.assertNotIn(banned, code)


class RestingScreenTests(unittest.TestCase):
    """The resting screen: nothing starts until someone taps, and it comes back once everyone has
    been gone for RESTING_AFTER_SECONDS."""

    def test_comes_back_after_five_seconds_with_nobody_in_view(self):
        self.assertEqual(system.RESTING_AFTER_SECONDS, 5)
        self.assertFalse(system.back_to_resting(0.0))
        self.assertFalse(system.back_to_resting(4.9))
        self.assertTrue(system.back_to_resting(5.0))
        with patch.object(system, "USE_RESTING_SCREEN", False):
            self.assertFalse(system.back_to_resting(60.0))

    def test_nobody_gets_a_filter_or_the_intro_while_resting(self):
        # main.py holds everyone back the same way the schedule does while the resting screen is up
        with patch.object(system, "DWELL_SECONDS", 0.0), patch.dict(system.FORWARD, yaw=0.0, pitch=0.0), \
                patch("viraj.emoji_filter.RENDERER", "software"), contextlib.redirect_stdout(io.StringIO()):
            effect = RecordingFilter(system.load_filters(["clown_nose"])[0])
            faces = {}
            for t in np.arange(0, 4.01, 0.1):
                AcceptanceTests.step(self, faces, effect, 10 + t, [detection()], off=True)
            self.assertEqual(effect.calls, [])
            self.assertTrue(all(f.eligible_since is None and not f.opted_in for f in faces.values()))

    def test_dashboard_accepts_start_with_or_without_a_filter(self):
        import json
        from common import dashboard_server
        self.assertIn("start", dashboard_server.ACTIONS)
        handler = dashboard_server._Handler.__new__(dashboard_server._Handler)
        sent = []
        handler.dashboard = SimpleNamespace(commands=SimpleNamespace(put=sent.append))
        handler.path = "/api/do"
        handler._send = lambda code, *a, **k: sent.append(code)
        for body in ({"action": "start"}, {"action": "start", "value": 2}, {"action": "start", "value": "x"}):
            raw = json.dumps(body).encode()
            handler.headers = {"Content-Length": str(len(raw))}
            handler.rfile = io.BytesIO(raw)
            handler.do_POST()
        self.assertEqual(sent, [("start", None), 204, ("start", 2), 204, 400])


class DashboardStateTests(unittest.TestCase):
    def test_dashboard_state_has_no_duplicate_keys(self):
        # A repeated key silently replaces the first one; "active" once got overwritten this way,
        # which sent the wheel to the wrong filter.
        import ast
        with open(system.__file__) as source:
            tree = ast.parse(source.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
                if "filters" in keys and "active" in keys:
                    self.assertEqual(len(keys), len(set(keys)), keys)
                    return
        self.fail("dashboard state not found")
