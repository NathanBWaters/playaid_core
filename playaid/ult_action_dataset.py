import cv2
import os

import imutils
import numpy as np
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
import csv
from collections import defaultdict

from playaid.constants import (
    CHAR_LIST,
    REPLAYS_DIR,
    GROUND_TRUTH_VIDEO,
    GROUND_TRUTH_SAMPLE,
    ACTION_RECOG_NUM_FRAMES_PER_SAMPLE,
    ACTION_RECOG_FRAME_DELTA,
    GROUND_TRUTH_DIR,
    ACTION_GROUND_TRUTH_TRAIN,
    ACTION_GROUND_TRUTH_VAL,
    ACTION_GROUND_TRUTH_TEST,
)
from playaid.timeline import cache_dataset
from playaid.fighter import Fighter, YoloCrop
from playaid.anim_ontology import MOVE_TO_CLASS_ID, FIGHTER_ENUM_TO_NAME
from playaid.dataset_utils import (
    get_character_actions_animations_dict,
    get_stage_paths,
    augment_synth_char_crop,
    augment_char_crop,
    action_sample_from_frame,
    action_sample_from_frame_middle_out,
)


def random_crop_pil_image(img, x, y):
    """ """
    width, height = img.size
    x1 = random.randrange(0, width - x)
    y1 = random.randrange(0, height - y)
    return img.crop((x1, y1, x1 + x, y1 + y)), (x1, y1)


def clamp(minimum, x, maximum):
    return max(minimum, min(x, maximum))


def slightly_move_crop_pil_image(img, x, y, upper_left, range):
    """ """
    width, height = img.size
    x_offset = random.randrange(-range, range)
    y_offset = random.randrange(-range, range)
    x1 = clamp(0, upper_left[0] + x_offset, width - x)
    y1 = clamp(0, upper_left[1] + y_offset, height - y)
    return img.crop((x1, y1, x1 + x, y1 + y)), (x1, y1)


def select_frame_path_from_char_anim_dict(char_anim_dict: dict, fighter_name: str, animations):
    """ """
    action = random.choice(animations)

    if action not in char_anim_dict[fighter_name]:
        bad_actions = [a for a in animations if a not in char_anim_dict[fighter_name]]
        raise Exception(
            f"Requested action {action} which is not available locally. Other actions not "
            + f"available are: {bad_actions}"
        )

    if not char_anim_dict[fighter_name][action].keys():
        raise Exception(f"Requested action {action} for {fighter_name} has no associated body")

    body_type = random.choice(list(char_anim_dict[fighter_name][action].keys()))

    if not char_anim_dict[fighter_name][action][body_type].keys():
        raise Exception(
            f"Requested action {action} for {fighter_name} and body {body_type} has no raw anim"
        )

    raw_anim_name = random.choice(list(char_anim_dict[fighter_name][action][body_type].keys()))
    if not char_anim_dict[fighter_name][action][body_type][raw_anim_name].keys():
        raise Exception(
            f"Requested raw action {raw_anim_name} for {fighter_name} and body {body_type} has no cam"
        )

    cam = random.choice(list(char_anim_dict[fighter_name][action][body_type][raw_anim_name].keys()))
    if not char_anim_dict[fighter_name][action][body_type][raw_anim_name][cam]:
        raise Exception(
            f"Requested cam for {fighter_name} {body_type} {raw_anim_name} {cam} has no anim"
        )

    all_frames = char_anim_dict[fighter_name][action][body_type][raw_anim_name][cam]
    return random.choice(all_frames)


def load_and_augment_frame_to_stage_crop(frame_path: str, stage_crop, synth_difficulty):
    """ """
    # We have to copy it so we don't keep pasting over the original
    stage_crop = stage_crop.copy()
    width, height = stage_crop.size
    char_frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
    if synth_difficulty:
        synth_difficulty_map = {
            1: {
                "horizontal_flip": 0.0,
                "hard_mode": 0.0,
                "downscale": 0.1,
                "resize": 0.1,
            },
            2: {
                "horizontal_flip": 0.0,
                "hard_mode": 0.2,
                "downscale": 0.3,
                "resize": 0.3,
            },
        }
        char_frame = augment_synth_char_crop(char_frame, **synth_difficulty_map[synth_difficulty])
    # TODO: make it not take up the whole frame every time.
    if char_frame.shape[0] > char_frame.shape[1]:
        char_frame = imutils.resize(char_frame, height=height)
    else:
        char_frame = imutils.resize(char_frame, width=width)

    char_frame = cv2.cvtColor(char_frame, cv2.COLOR_BGRA2RGBA)
    char_frame = Image.fromarray(char_frame)
    paste_x = int((stage_crop.width - char_frame.width) / 2)
    paste_y = int((stage_crop.height - char_frame.height) / 2)

    if synth_difficulty:
        paste_x += random.randint(-40, 40)
        paste_y += random.randint(-40, 40)

    # Paste it mostly in the middle.
    stage_crop.paste(char_frame, (paste_x, paste_y), char_frame)
    return stage_crop


class UltActionRecogDataset(Dataset):
    def __init__(
        self,
        split,
        num_samples,
        img_dimension,
        anim_subset,
        num_frames_per_sample=ACTION_RECOG_NUM_FRAMES_PER_SAMPLE,
        frame_delta=ACTION_RECOG_FRAME_DELTA,
        char_subset=[],
        randomize_stage_background=False,
        move_stage_background=False,
        ground_truth_offset=0,
        synth_difficulty=0,
        manual_ground_truth_csv=GROUND_TRUTH_SAMPLE,
        train_synth_rate=0.7,
        num_preceding_actions=8,
        crop_size=128,
    ):
        """
        @param anim_subset: the list of animations that will be returned. If empty, all animations
        will be returned.
        @param num_frames_per_sample: either an integer or an array. If an array, it will select
        one from it until switch_num_frames_per_sample() is called
        @param train_synth_rate: percent of the time that the training data will be synthetic vs
        real.
        """
        self.split = split
        self.num_samples = num_samples
        self.crop_size = crop_size
        # self.char_anim_dict = get_character_actions_animations_dict()
        # Broke this by deleting the dataset locally...
        self.char_anim_dict = {}
        self.num_frames_per_sample = (
            num_frames_per_sample
            if isinstance(num_frames_per_sample, int)
            else random.choice(num_frames_per_sample)
        )
        self.num_frames_per_sample_options = (
            [num_frames_per_sample]
            if isinstance(num_frames_per_sample, int)
            else num_frames_per_sample
        )
        # Distance between the frames. So n-frame_delta, n, n+frame_delta.
        self.frame_deltas = frame_delta if type(frame_delta) == list else [frame_delta]
        self.stage_paths = get_stage_paths()
        self.img_dimension = img_dimension
        self.animations = anim_subset
        self.characters = char_subset if char_subset else CHAR_LIST
        self.randomize_stage_background = randomize_stage_background
        self.move_stage_background = move_stage_background
        self.num_preceding_actions = num_preceding_actions

        self.ground_truth_offset = ground_truth_offset
        self.synth_difficulty = synth_difficulty
        self.manual_ground_truth_csv = manual_ground_truth_csv
        self.train_synth_rate = train_synth_rate

        self.training_video_to_sample, self.training_move_to_frame = cache_dataset(
            ACTION_GROUND_TRUTH_TRAIN, self.characters
        )
        self.val_video_to_sample, self.val_move_to_frame = cache_dataset(
            ACTION_GROUND_TRUTH_VAL, self.characters
        )
        self.test_video_to_sample, self.test_move_to_frame = cache_dataset(
            ACTION_GROUND_TRUTH_TEST, self.characters
        )

        with open(self.manual_ground_truth_csv) as output_file:
            # Split the ground truth between train, val, and test.
            num_lines = len(output_file.readlines())
            lines = [i for i in range(num_lines)]
            train_lines = lines[0 : int(num_lines / 3)]
            val_lines = lines[int(num_lines / 3) : int(num_lines / 3 * 2)]
            test_lines = lines[int(num_lines / 3 * 2) :]
            (
                self.train_ground_truth_labels,
                self.train_action_to_frames,
            ) = self.load_ground_truth_labels(train_lines)
            (
                self.val_ground_truth_labels,
                self.val_action_to_frames,
            ) = self.load_ground_truth_labels(val_lines)
            (
                self.test_ground_truth_labels,
                self.test_action_to_frames,
            ) = self.load_ground_truth_labels(test_lines)

        # Mark that we haven't loaded in the replay files yet.
        self.loaded_replays = False

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns a triplet pair of frames for an animation as input. The corresponding label matches
        the middle frame. The label is made up of two parts: the character and the action.
        """
        if self.split == "simple":
            return self.simple_dataset(idx)
        if self.split == "train":
            return self.ground_truth(
                self.training_video_to_sample, self.training_move_to_frame, idx
            )
        elif self.split == "validation":
            return self.ground_truth(self.val_video_to_sample, self.val_move_to_frame, idx)
        else:
            return self.ground_truth(self.test_video_to_sample, self.test_move_to_frame, idx)

    def ground_truth(self, video_to_sample, move_to_frame, idx):
        """For now, loads a single video and its corresponding ground truth"""
        # Choose a random character.
        # fighter_name = random.choice(list(ground_truth_labels.keys()))
        if not list(video_to_sample.keys()):
            print("video_to_sample is empty!")

        # Choose a fighter randomly, then choose a move randomly.  From that, choose a frame frame.
        fighter_name = random.choice(list(move_to_frame.keys()))
        action_name = random.choice(list(move_to_frame[fighter_name].keys()))
        video_name, selected_frame = random.choice(move_to_frame[fighter_name][action_name])

        frame_delta = random.choice(self.frame_deltas)
        max_frames = len(video_to_sample[video_name][fighter_name])

        frame_nums = action_sample_from_frame_middle_out(
            selected_frame,
            self.num_frames_per_sample,
            frame_delta,
            min_frame=0,
            max_frames=max_frames,
            clamp=True,
        )

        frames = []
        actions = []
        frame_paths = []

        # Also append the preceding N actions.
        preceding_actions = []
        for i in range(
            selected_frame - self.num_preceding_actions, selected_frame - self.num_preceding_actions
        ):
            frame_num = i
            if frame_num < 0:
                frame_num = 0

            frame_path, label_path = video_to_sample[video_name][fighter_name][frame_num]
            with open(label_path, "r") as file:
                action_string = file.read()

            preceding_actions.append(action_string)

        preceding_actions_label = [
            self.animations.index(action)
            if action in self.animations
            else self.animations.index("Unknown")
            for action in preceding_actions
        ]

        for frame_num in frame_nums:
            frame_path, label_path = video_to_sample[video_name][fighter_name][frame_num]
            frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = imutils.resize(frame, width=self.crop_size)

            if self.synth_difficulty:
                synth_difficulty_map = {
                    1: {
                        "horizontal_flip": 0.0,
                        "hard_mode": 0.0,
                        "downscale": 0.1,
                        "resize": 0.4,
                        "course_dropout": 0.9,
                        "channel_dropout": 0.0,
                        "pixel_dropout": 0.1,
                        "gauss_noise": 0.4,
                    },
                    2: {
                        "horizontal_flip": 0.0,
                        "hard_mode": 0.2,
                        "downscale": 0.3,
                        "resize": 0.3,
                        "course_dropout": 0.2,
                        "channel_dropout": 0.01,
                        "pixel_dropout": 0.1,
                        "gauss_noise": 0.8,
                    },
                }
                frame = augment_char_crop(
                    frame, **synth_difficulty_map[self.synth_difficulty], output_size=self.crop_size
                )

            assert frame.shape == (
                self.crop_size,
                self.crop_size,
                3,
            ), (
                f"Frame does not have correct shape, expected {self.crop_size}x{self.crop_size}x3"
                + f"but got f{frame.shape}"
            )

            with open(label_path, "r") as file:
                action_string = file.read()

            actions.append(action_string)
            # These should all be the correct dimension.
            frames.append(frame)
            frame_paths.append(frame_path)

        input_frames = torch.tensor(np.array(frames))
        input_frames = input_frames.permute(0, 3, 1, 2)
        # Have an "unknown" label be the last one
        anim_label = [
            self.animations.index(action)
            if action in self.animations
            else self.animations.index("Unknown")
            for action in actions
        ]
        return (
            input_frames.float() / 255.0,
            torch.tensor(self.characters.index(fighter_name)),
            torch.tensor(anim_label),
            {
                "char": fighter_name,
                "frames": [np.array(f) for f in frames],
                "frame_paths": [os.path.basename(f) for f in frame_paths],
                "actions": actions,
                "frame_delta": frame_delta,
                "preceding_actions": preceding_actions,
                "preceding_actions_tensor": torch.tensor(preceding_actions_label),
            },
        )

    def simple_dataset(self, idx):
        """
        Tests the RNN aspect of our architecture by only returning two different sets of frames:
        - forward smash
        - up-air.

        In the middle of both sets is a stray frame.  We want the model to predict for that frame
        the surrounding actions.  If it can do that, then it can learn one frame influences the
        next.
        """
        center_frame = self.char_anim_dict["byleth"]["ForwardSmash"]["c00"]["c03attacks4hi_frame"][
            "90"
        ][50]

        batch_1 = [
            self.char_anim_dict["byleth"]["NeutralAir"]["c00"]["c05attackairn_frame"]["90"][30],
            center_frame,
            self.char_anim_dict["byleth"]["NeutralAir"]["c00"]["c05attackairn_frame"]["90"][35],
        ]
        batch_2 = [
            self.char_anim_dict["byleth"]["BackAir"]["c00"]["c05attackairb_frame"]["90"][20],
            center_frame,
            self.char_anim_dict["byleth"]["BackAir"]["c00"]["c05attackairb_frame"]["90"][25],
        ]

        selected_batch = batch_1 if idx % 2 else batch_2
        actions = ["NeutralAir"] * 3 if idx % 2 else ["BackAir"] * 3
        anim_label = [
            self.animations.index(action) if action in actions else self.animations.index("Unknown")
            for action in actions
        ]
        stage_path = self.stage_paths[0]
        stage = Image.open(stage_path)
        stage_cropped = stage.crop((0, 0, self.img_dimension, self.img_dimension))

        frames = []
        for f in selected_batch:
            frames.append(
                np.array(load_and_augment_frame_to_stage_crop(f, stage_cropped, synth_difficulty=0))
            )

        input_frames = torch.tensor(np.array(frames))
        input_frames = input_frames.permute(0, 3, 1, 2)

        return (
            input_frames.float() / 255.0,
            torch.tensor(self.characters.index("byleth")),
            torch.tensor(anim_label),
            {
                "char": "byleth",
                "frames": [np.array(f) for f in frames],
                "frame_paths": [f"{n}.png" for n in range(len(selected_batch))],
                "actions": actions,
            },
        )

    def manual_ground_truth(self, idx, ground_truth_labels, action_to_frames):
        """
        Returns the data for a manually annotated video of MKLeo vs Tweek.
        """
        # Choose a random character.
        # fighter_name = random.choice(list(ground_truth_labels.keys()))
        fighter_name = random.choice(self.characters)
        selected_action = random.choice(
            [
                a
                for a in self.animations
                # We have to make sure the animation actually has frames
                if a != "Unknown" and action_to_frames[fighter_name][a]
            ]
        )
        last_actual_frame = random.choice(action_to_frames[fighter_name][selected_action])
        # last_valid_frame = self.frame_to_valid_frame[last_actual_frame]

        frame_nums = action_sample_from_frame(
            last_actual_frame,
            self.num_frames_per_sample,
            self.frame_deltas,
            list(ground_truth_labels[fighter_name].keys()),
        )

        cap = cv2.VideoCapture(GROUND_TRUTH_VIDEO)
        frames = []

        actions = []
        fighters = []
        actual_frame_nums = []
        for actual_frame_num in frame_nums:
            # actual_frame_num = self.valid_frames_to_frame[valid_frame_num]
            actual_frame_nums.append(actual_frame_num)
            cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame_num)
            res, frame = cap.read()
            if not res:
                print("Hmmm")
            assert res, f"requested invalid frame {actual_frame_num} from ground truth "
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Cut out crop from the frame.
            (
                frame_num,
                fighter_name,
                action,
                center_x,
                center_y,
                crop_width,
                crop_height,
            ) = ground_truth_labels[fighter_name][actual_frame_num]
            fighter = Fighter(
                frame_num,
                fighter_name,
                action,
                action_confidence=1.0,
                crop=YoloCrop(center_x, center_y, crop_width, crop_height),
            )
            actions.append(action)
            res, cropped_frame = fighter.crop.square_crop(frame)
            frames.append(cropped_frame)
            fighters.append(fighter)

        input_frames = torch.tensor(np.array(frames))
        input_frames = input_frames.permute(0, 3, 1, 2)
        # Have an "unknown" label be the last one
        anim_label = [
            self.animations.index(action)
            if action in self.animations
            else self.animations.index("Unknown")
            for action in actions
        ]
        return (
            input_frames.float() / 255.0,
            torch.tensor(self.characters.index(fighter_name)),
            torch.tensor(anim_label),
            {
                "char": fighter_name,
                "frames": [np.array(f) for f in frames],
                "frame_paths": [f"{f.frame_num}.png" for f in fighters],
                "actions": actions,
            },
        )

    def load_ground_truth_labels(self, lines):
        """ """
        labels = defaultdict(dict)
        action_to_frames = {}
        # Maps frames that correspond to the animation subset with their actual frames.
        # TODO: actually use these.
        valid_frames_to_frame = {}
        frames_to_valid_frame = {}
        i = 0
        with open(self.manual_ground_truth_csv) as output_file:
            reader = csv.reader(output_file)
            for row in reader:
                # Make sure we only get ground truth from the corresponding lines.
                if reader.line_num == 1 or reader.line_num not in lines:
                    # Skip the header
                    continue

                frame_num = int(row[0])
                fighter_name = row[1]
                action = row[2]

                # if action not in self.animations:
                #     continue

                center_x = float(row[3])
                center_y = float(row[4])
                crop_width = float(row[5])
                crop_height = float(row[6])
                labels[fighter_name][frame_num] = (
                    frame_num,
                    fighter_name,
                    action,
                    center_x,
                    center_y,
                    crop_width,
                    crop_height,
                )
                if fighter_name not in action_to_frames:
                    action_to_frames[fighter_name] = defaultdict(list)

                action_to_frames[fighter_name][action].append(frame_num)
                valid_frames_to_frame[i] = frame_num
                frames_to_valid_frame[frame_num] = i
                i += 1
        return (
            dict(labels),
            dict(action_to_frames),
        )

    def make_synth_more_challenging(self):
        if self.synth_difficulty < 2:
            self.synth_difficulty += 1
            print(f"INCREASING SYNTHETIC DIFFICULTY TO {self.synth_difficulty}")

    def switch_num_frames_per_sample(self):
        self.num_frames_per_sample = random.choice(self.num_frames_per_sample_options)

    def get_synth(self, idx):
        char = random.choice(self.characters)
        char_label = self.characters.index(char)

        # Choose a random action as a stand-in to get the body.
        temp_action = random.choice([a for a in self.animations if a != "Unknown"])
        if not self.char_anim_dict[char][temp_action].keys():
            raise Exception(f"Requested action {temp_action} for {char} has no associated body")

        body_type = random.choice(list(self.char_anim_dict[char][temp_action].keys()))

        if not self.char_anim_dict[char][temp_action][body_type].keys():
            raise Exception(
                f"Requested action {temp_action} for {char} and body {body_type} has no raw anim"
            )

        # Concatenates more than one action together as a clip.
        mini_timeline_frames = []
        mini_timeline_actions = []
        i = 0
        while i < 2 or len(mini_timeline_frames) < self.num_frames_per_sample:
            # Now choose an actual action
            action = None
            while not action:
                selected_action = random.choice(self.animations)
                if selected_action == "Unknown":
                    action = random.choice(
                        list(set(self.char_anim_dict[char].keys()) - set(self.animations))
                    )

                elif selected_action in self.char_anim_dict[char]:
                    action = selected_action

                # Otherwise, keep trying

                # bad_actions = [
                #     a for a in self.animations if a not in self.char_anim_dict[char]
                # ]
                # raise Exception(
                #     f"Requested action {action} which is not available locally. Other actions not "
                #     + f"available are: {bad_actions}"
                # )

            raw_anim_name = random.choice(list(self.char_anim_dict[char][action][body_type].keys()))
            if not self.char_anim_dict[char][action][body_type][raw_anim_name].keys():
                raise Exception(
                    f"Requested raw action {raw_anim_name} for {char} and body {body_type} has no cam"
                )
            cam = random.choice(
                list(self.char_anim_dict[char][action][body_type][raw_anim_name].keys())
            )
            if not self.char_anim_dict[char][action][body_type][raw_anim_name][cam]:
                raise Exception(
                    f"Requested cam for {char} {body_type} {raw_anim_name} {cam} has no anim"
                )

            animation_frames = self.char_anim_dict[char][action][body_type][raw_anim_name][cam]

            mini_timeline_frames.extend(animation_frames)
            mini_timeline_actions.extend([selected_action] * len(animation_frames))

            i += 1

        num_frames = len(mini_timeline_frames)
        last_frame = random.randint(self.num_frames_per_sample, num_frames - 1)
        # TODO: we want to skip each frame.
        clip_frame_paths = mini_timeline_frames[
            last_frame - self.num_frames_per_sample : last_frame
        ]
        clip_actions = mini_timeline_actions[last_frame - self.num_frames_per_sample : last_frame]

        # frame_nums = action_sample_from_frame(
        #     last_frame,
        #     self.num_frames_per_sample,
        #     # Since synth is 60fps and real is 30fps, just multiply by 2.
        #     self.frame_deltas * 2,
        #     len(animation_frames),
        #     clamp=False,
        # )

        stage_path = random.choice(self.stage_paths)
        stage = Image.open(stage_path)
        stage_cropped, ul = random_crop_pil_image(stage, self.img_dimension, self.img_dimension)

        frames = []

        for frame_path in clip_frame_paths:
            if self.randomize_stage_background:
                stage_path = random.choice(self.stage_paths)
                stage = Image.open(stage_path)
                stage_cropped, ul = random_crop_pil_image(
                    stage, self.img_dimension, self.img_dimension
                )
            if self.move_stage_background and not self.randomize_stage_background:
                stage_cropped, ul = slightly_move_crop_pil_image(
                    Image.open(stage_path),
                    self.img_dimension,
                    self.img_dimension,
                    ul,
                    10,
                )

            frame = load_and_augment_frame_to_stage_crop(
                frame_path, stage_cropped, self.synth_difficulty
            )
            frames.append(frame)

        input_frames = torch.tensor(np.array([np.array(f) for f in frames]))
        input_frames = input_frames.permute(0, 3, 1, 2)
        anim_label = [self.animations.index(action) for action in clip_actions]
        return (
            input_frames.float() / 255.0,
            torch.tensor(char_label),
            torch.tensor(anim_label),
            {
                "char": char,
                "frames": [np.array(f) for f in frames],
                "frame_paths": clip_frame_paths,
                "actions": clip_actions,
            },
        )


if __name__ == "__main__":
    loader = UltActionRecogDataset("train", 32, 256, num_frames_per_sample=7, frame_delta=3)

    output1 = loader[0]

    print("🎉")
