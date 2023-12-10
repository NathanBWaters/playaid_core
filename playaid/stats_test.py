import unittest

from playaid.fighter import Fighter, YoloCrop
from playaid.stats import Stats


class TestStats(unittest.TestCase):
    def setUp(self):
        self.byleth = Fighter(
            frame_num=0,
            fighter_id=0,
            fighter_name="byleth",
            crop=YoloCrop(0.25, 0.25, 0.25, 0.25),
        )
        self.diddy = Fighter(
            frame_num=0,
            fighter_id=1,
            fighter_name="Diddy Kong",
            crop=YoloCrop(0.25, 0.25, 0.25, 0.25),
        )
        self.stats = Stats()

    def test_record_frame(self):
        self.byleth.action = "ForwardSmash"
        self.diddy.action = "UpTilt"
        self.stats.record_frame([self.byleth, self.diddy])
        self.assertDictEqual(
            dict(self.stats.stats),
            {
                0: {
                    "action_count": {"ForwardSmash": 1},
                    "latest_action": "ForwardSmash",
                    "latest_action_frame": 0,
                },
                1: {
                    "action_count": {"UpTilt": 1},
                    "latest_action": "UpTilt",
                    "latest_action_frame": 0,
                },
            },
        )

        self.byleth.frame_num = 2
        self.diddy.frame_num = 2
        self.stats.record_frame([self.byleth, self.diddy])
        self.assertDictEqual(
            dict(self.stats.stats),
            {
                0: {
                    "action_count": {"ForwardSmash": 1},
                    "latest_action": "ForwardSmash",
                    "latest_action_frame": 0,
                },
                1: {
                    "action_count": {"UpTilt": 1},
                    "latest_action": "UpTilt",
                    "latest_action_frame": 0,
                },
            },
        )

        # Record a damage instace
        self.byleth.frame_num = 3
        self.diddy.frame_num = 3
        self.diddy.action = "Damaged"
        self.stats.record_frame([self.byleth, self.diddy])
        self.assertDictEqual(
            dict(self.stats.stats),
            {
                0: {
                    "action_count": {"ForwardSmash": 1},
                    "successful_action_count": {"ForwardSmash": 1},
                    "latest_action": "ForwardSmash",
                    "latest_action_frame": 0,
                },
                1: {
                    "action_count": {"UpTilt": 1, "Damaged": 1},
                    "latest_action": "Damaged",
                    "latest_action_frame": 3,
                },
            },
        )


if __name__ == "__main__":
    unittest.main()
