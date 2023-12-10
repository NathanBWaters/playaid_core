import os
import unittest
import numpy as np

from playaid.timeline import GTVideo
from playaid.constants import GROUND_TRUTH_DIR


class TestTimeline(unittest.TestCase):
    def setUp(self):
        video_path = os.path.join(GROUND_TRUTH_DIR, "byleth_v_diddy_1/byleth_v_diddy_1.mp4")
        label_path = os.path.join(GROUND_TRUTH_DIR, "byleth_v_diddy_1/output.log")
        self.video = GTVideo(video_path, label_path)

    def test_get_fps(self):
        fps = self.video.get_fps()
        self.assertIsInstance(fps, float)

    def test_get_frame(self):
        frame = self.video.get_frame(0)
        self.assertIsNotNone(frame)  # more specific assertion depends on frame type

    def test_get_frames(self):
        frames = self.video.get_frames(0, 5)
        self.assertEqual(len(frames), 5)

        # Check that frame is a numpy array
        self.assertIsInstance(frames[0], np.ndarray)

        # Check that frame is not empty
        self.assertTrue(frames[0].any())

    def test_load_ground_truth(self):
        self.video.load_ground_truth()
        self.assertTrue(bool(self.video.ground_truth))  # Check ground truth is not empty

    def test_get_ground_truth(self):
        ground_truth = self.video.get_ground_truth(0, 5)
        self.assertEqual(len(ground_truth), 5)

    def test_get_frames_and_labels(self):
        frames, ground_truth = self.video.get_frames_and_labels(0, 5)
        self.assertEqual(len(frames), 5)
        self.assertEqual(len(ground_truth), 5)


if __name__ == "__main__":
    unittest.main()
