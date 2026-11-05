"""Deterministic acceptance replays, without a camera, model download or GPIO.

These verify shared decisions and artwork against synthetic measured geometry.
Live detector accuracy and physical pass rates require the trials in ACCEPTANCE.md.
"""
import contextlib
from datetime import datetime
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
from viraj.emoji_filter import rotation


def detection(distance=24, center=(320, 240), roll=0, yaw=0, pitch=0):
    """Generate current-frame points at a calibrated distance, including iris mesh."""
    size = system.FOCAL_PX * system.EYE_GAP_INCHES / distance
    angle = math.radians(roll)
    rotate = np.array([[math.cos(angle), -math.sin(angle)],
                       [math.sin(angle), math.cos(angle)]])
    turn = math.cos(math.radians(yaw))

    def pixels(points):
        return (np.asarray(points) * [turn, 1]) @ rotate.T * size + center

    points = pixels([[-0.5, -0.55], [0.5, -0.55], [0, 0], [-0.4, 0.6], [0.4, 0.6]])
    row = np.array([center[0]-1.3*size, center[1]-1.5*size, 2.6*size, 3*size,
                    *points.ravel(), 1.0], dtype=np.float32)
    mesh = np.tile(np.asarray(center, dtype=np.float32), (478, 1))
    for eye_x, (contour, iris, ring) in zip((-0.5, 0.5), EYES):
        t = np.arange(16) * 2 * math.pi / 16
        mesh[list(contour)] = pixels(np.column_stack((eye_x-0.24*np.cos(t), -0.55+0.12*np.sin(t))))
        mesh[iris] = pixels([eye_x, -0.55])
        mesh[list(ring)] = pixels([[eye_x+0.09, -0.55], [eye_x, -0.64],
                                  [eye_x-0.09, -0.55], [eye_x, -0.46]])
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

    def step(self, faces, effect, now, detections=(), *, off=False, sensor=None):
        next_id = [max(faces, default=-1)+1]
        system.track(faces, detections, now, next_id, (640, 480))
        frame = np.full((480, 640, 3), 140, np.uint8)
        result = system.process_faces(frame, faces, [effect], effect, now, 0.1,
                                      filters_off=off, sensor_inches=sensor, draw_notes=False)
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
                self.assertIsInstance(filters[0], AstronautFilter)

    def test_short_child_seated_and_45_degree_tilt_ten_trials_per_filter(self):
        for actual in self.filters:
            for trial in range(10):
                # Height affects image placement, never eligibility. Test both roll directions.
                for center, roll in [((260, 300), 0), ((320, 240), (-1)**trial*45), ((420, 350), 0)]:
                    with self.subTest(filter=actual.name, trial=trial, center=center, roll=roll):
                        faces, effect = {}, RecordingFilter(actual)
                        for now in np.linspace(10, 13.1, 32):
                            frame, _ = self.step(faces, effect, float(now), [detection(30, center, roll)])
                        self.assertTrue(effect.calls)
                        self.assertAlmostEqual(faces[0].roll, roll)
                        self.assertFalse(np.all(frame == 140), "actual renderer must draw")

    def test_every_visit_shorter_than_three_seconds_is_invisible(self):
        for actual in self.filters:
            for trial in range(10):
                with self.subTest(filter=actual.name, trial=trial):
                    faces, effect = {}, RecordingFilter(actual)
                    for elapsed in np.linspace(0, 2.999, 16):
                        # Fast lateral crossing right inside the zone, facing forward.
                        center = (80+480*elapsed/3, 240)
                        frame, _ = self.step(faces, effect, 10+elapsed, [detection(24, center)])
                        self.assertTrue(np.all(frame == 140))
                    self.step(faces, effect, 13.01)
                    self.assertEqual(effect.calls, [])

    def test_sixty_degree_turn_starts_at_point_one_and_increases_five_trials(self):
        for actual in self.filters:
            for trial in range(5):
                with self.subTest(filter=actual.name, trial=trial):
                    faces, effect = {}, RecordingFilter(actual)
                    det = detection(yaw=(-1)**trial*60)
                    for now in np.arange(10, 13, 0.1):
                        self.step(faces, effect, float(now), [det])
                    self.assertFalse(effect.calls)
                    for now in (13.0, 13.1, 13.2, 13.4):
                        self.step(faces, effect, now, [det])
                    opacity = [value for _, value in effect.calls]
                    self.assertAlmostEqual(opacity[0], 0.1)
                    self.assertTrue(all(a < b for a, b in zip(opacity, opacity[1:])))
                    self.assertAlmostEqual(opacity[-1], 1)

    def test_profile_five_feet_two_second_crossing_five_trials(self):
        for actual in self.filters:
            for trial in range(5):
                faces, effect = {}, RecordingFilter(actual)
                for elapsed in np.linspace(0, 2, 21):
                    self.step(faces, effect, 10+elapsed, [detection(60, (80+240*elapsed, 240), yaw=90)])
                self.assertFalse(effect.calls, actual.name)

    def test_in_zone_profile_and_invalid_pose_never_draw_even_after_dwell(self):
        for actual in self.filters:
            for pose in [(90, 0, 0), (0, 90, 0), (0, 0, 90), (float("nan"), 0, 0), (60.01, 0, 0)]:
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
            for now, det in [(13.5, detection(30.01)), (13.6, detection(yaw=90))]:
                calls = len(effect.calls)
                self.step(faces, effect, now, [det])
                self.assertEqual(len(effect.calls), calls)
                self.assertEqual(faces[0].opacity, 0)

    def test_schedule_exact_cutoff_new_and_existing_visitors(self):
        for stamp, expected in [("16:14:59.999999", False), ("16:15:00", True),
                                ("16:15:00.000001", True), ("23:59:59", True), ("00:00:00", False)]:
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

    def test_interruptions_require_fresh_dwell_and_cleanup(self):
        for actual in self.filters:
            faces, effect = {}, RecordingFilter(actual)
            for now in np.arange(10, 12.9, 0.1):
                self.step(faces, effect, float(now), [detection()])
            self.step(faces, effect, 12.9)  # missed detection resets pending activation
            for now in np.arange(13, 16, 0.1):
                self.step(faces, effect, float(now), [detection()])
            self.assertFalse(effect.calls)
            self.step(faces, effect, 16.1, [detection()])
            self.assertTrue(effect.calls)
            self.step(faces, effect, 16.7)
            self.assertEqual(faces, {})
            self.assertEqual(effect.forgotten, [0])

    def test_fade_toggle_cannot_bypass_passerby_dwell(self):
        with patch.object(system, "USE_FADE_IN", False):
            face = system.Face(0)
            for now in (10, 11, 12.999):
                system.update_opacity(face, True, now, 1)
                self.assertEqual(face.opacity, 0)
            system.update_opacity(face, True, 13, 0.001)
            self.assertEqual(face.opacity, 1)

    def test_independent_multi_user_dwell(self):
        for actual in self.filters:
            faces, effect = {}, RecordingFilter(actual)
            first, second = detection(28, (180, 260)), detection(24, (460, 340), roll=-45)
            for now in np.linspace(10, 13.1, 32):
                dets = [first] if now < 11 else [first, second]
                self.step(faces, effect, float(now), dets)
            self.assertEqual({fid for fid, _ in effect.calls}, {0})
            for now in np.arange(13.1, 14.3, 0.1):
                self.step(faces, effect, float(now), [first, second])
            self.assertEqual({fid for fid, _ in effect.calls}, {0, 1})

    def test_sensor_distance_uses_the_same_zone(self):
        for actual in self.filters:
            faces, effect = {}, RecordingFilter(actual)
            for now in np.linspace(10, 13.1, 32):
                self.step(faces, effect, float(now), [detection(60)], sensor=30)
            self.assertTrue(effect.calls)
            calls = len(effect.calls)
            self.step(faces, effect, 13.2, [detection(60)], sensor=31)
            self.assertEqual(len(effect.calls), calls)

    def test_tracking_and_resizing_five_feet_to_six_inches_five_trials(self):
        nose = ClownNoseFilter()
        for trial in range(5):
            faces, effect = {}, RecordingFilter(nose)
            widths = []
            for index, distance in enumerate((60, 48, 36, 30, 24, 18, 12, 6)):
                now = 10 + index*0.1
                self.step(faces, effect, now, [detection(distance)])
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
            self.assertFalse(effect.calls, "approach has not yet spent three seconds in the zone")

    def test_sway_at_five_feet_has_no_anchor_lag_and_remains_hidden_five_trials(self):
        for actual in self.filters:
            for trial in range(5):
                faces, effect = {}, RecordingFilter(actual)
                # Include both roll and yaw interpretations of a 30-degree sway.
                for index, angle in enumerate((0, -30, 30, -30, 30, 0)):
                    det = detection(60, (320+angle, 240), roll=angle, yaw=angle)
                    self.step(faces, effect, 10+index*0.1, [det])
                    np.testing.assert_allclose(faces[0].nose, det[0][8:10], atol=1e-6)
                    self.assertAlmostEqual(faces[0].distance, 60, places=4)
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
        with patch.object(system, "datetime") as clock:
            clock.now.return_value = datetime(2026, 11, 5, 16, 15)
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
                frame, _ = self.step(faces, effect, float(now), [detection(6, (320, 350))])
            self.assertTrue(effect.calls)
            self.assertFalse(np.all(frame == 140))

    def test_helmet_visibly_rotates_and_preserves_face_window(self):
        helmet = AstronautFilter()
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
        self.assertTrue(np.all(original[ny, nx] == 0), "face remains visible through visor")
        # The off-center bottom of the helmet identifies the direction of rotation.
        mask = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) > 20
        rotated_mask = cv2.cvtColor(tilted, cv2.COLOR_BGR2GRAY) > 20
        y, x = np.where(mask)
        ry, rx = np.where(rotated_mask)
        anchor = np.mean(face.points[2:7], axis=0)
        expected = (np.array([x.mean(), y.mean()])-anchor) @ np.array([[2**-0.5, 2**-0.5],
                                                                 [-2**-0.5, 2**-0.5]]) + anchor
        np.testing.assert_allclose([rx.mean(), ry.mean()], expected, atol=2)

    def test_overlay_zero_opacity_and_offscreen_are_noops(self):
        art = premultiply(np.full((20, 20, 4), 255, np.uint8))
        for center, opacity in [((320, 240), 0), ((-1000, -1000), 1)]:
            frame = np.full((480, 640, 3), 100, np.uint8)
            draw_artwork(frame, art, center, 100, 45, opacity)
            self.assertTrue(np.all(frame == 100))


if __name__ == "__main__":
    unittest.main()
