import unittest

from playaid.fighter import Fighter  # replace with your file's name


class TestGTVideo(unittest.TestCase):
    def test_fighter(self):
        # byleth
        ground_truth = {
            # "camera_fov": 17.0,
            "camera_fov": 30.0,
            "camera_position": {
                "x": -0.00013416587898973376,
                "y": 14.01315975189209,
                "z": 167.240966796875,
            },
            "camera_target_position": {
                "x": -0.0001499500940553844,
                "y": 11.852787017822266,
                "z": 0.0,
            },
            "damage": 0.0,
            "facing": -1.0,
            "fighter_id": 0,
            "motion_kind": 19292652517,
            "num_frames_left": 25200,
            "pos_x": 27.0,
            "pos_y": 0.1,
            "shield_size": 50.0,
            "status_kind": 0,
            "stock_count": 3,
        }
        fighter = Fighter(frame_num=0, data=ground_truth)

        self.assertEqual(fighter.position_in_world, [27.0, 0.1, 0.0])
        self.assertEqual(fighter.damage, 0.0)
        self.assertEqual(fighter.facing, -1.0)
        self.assertEqual(fighter.fighter_id, 0)
        self.assertEqual(fighter.motion_kind, 19292652517)
        self.assertEqual(fighter.num_frames_left, 25200)
        self.assertEqual(fighter.pos_x, 27.0)
        self.assertEqual(fighter.pos_y, 0.1)
        self.assertEqual(fighter.shield_size, 50.0)
        self.assertEqual(fighter.status_kind, 0)
        self.assertEqual(fighter.stock_count, 3)

        # Now make sure the derived fields are also correct.
        self.assertEqual(fighter.action_string, "wait")
        self.assertEqual(fighter.action, "Wait")
        self.assertEqual(fighter.fighter_name, "byleth")
        self.assertEqual(fighter.char_class_id, 0)

        # self.assertEqual(fighter.crop, "Byleth")


if __name__ == "__main__":
    unittest.main()
