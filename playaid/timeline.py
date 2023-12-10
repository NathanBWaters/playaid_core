"""
A timeline is built up either from ground truth or predictions.
Currently, it's a simple array.
"""

import cv2
import json
import csv
import os
import glob
import yaml

from playaid.fighter import Fighter

# from playaid.stats import Stats
from playaid.anim_ontology import FIGHTER_NAME_TO_ENUM


def yield_fighters_and_stats(stats, video_path: str, label_path: str, log_offset=0):
    timeline = load_ground_truth_from_path(label_path, log_offset=log_offset)
    fighters = [Fighter(frame_num=0, data=json_data) for json_data in timeline[0]]
    max_frames = len(timeline)

    for i in range(max_frames):
        if i >= len(timeline):
            break
        fighters = update_fighters_from_timeline(i, timeline[i], fighters)
        stats.record_frame(fighters)
        yield (fighters, stats, i)


def yield_interval_fighters_and_stats(
    stats, interval: int, video_path: str, label_path: str, log_offset=0
):
    timeline = load_ground_truth_from_path(label_path, log_offset=log_offset)
    fighters = [Fighter(frame_num=0, data=json_data) for json_data in timeline[0]]
    max_frames = len(timeline)

    for i in range(max_frames):
        if i >= len(timeline):
            break

        fighters = update_fighters_from_timeline(i, timeline[i], fighters)
        stats.record_frame(fighters)

        if i % interval != 0 or i == 0:
            continue

        yield (fighters, stats, i)


def load_timeline_from_ai_output(file_path):
    timeline = []
    with open(file_path, "r") as f:
        ai_output = yaml.safe_load(f)

        max_frames = 600
        fighters = ["Joker", "Pikachu"]
        fighter_to_player_id = {
            "Pikachu": 0,
            "Joker": 1,
        }

        for i in range(max_frames):
            frame_data = []
            for fighter in fighters:
                fighter_data = ai_output[fighter][i]

                base = {
                    "raw_animation_frame_num": 0,
                    "attack_connected": False,
                    "camera_fov": 30.0,
                    "camera_position": {
                        "x": 0.0002484553260728717,
                        "y": 15.847139358520508,
                        "z": 148.460693359375,
                    },
                    "camera_target_position": {
                        "x": 0.0002776149194687605,
                        "y": 11.162917137145996,
                        "z": 0.0,
                    },
                    "can_act": True,
                    "damage": 0.0,
                    "facing": 1.0,
                    "fighter_id": fighter_to_player_id[fighter],
                    "fighter_name": FIGHTER_NAME_TO_ENUM[fighter],
                    "hitstun_left": 0.0,
                    "motion_kind": 19292652517,
                    "num_frames_left": 54000,
                    "pos_x": -50.0,
                    "pos_y": 0.21623137593269348,
                    "shield_size": 50.0,
                    "stage_id": 86,
                    "status_kind": 0,
                    "stock_count": 20,
                }

                # Now add the actual crop.
                base.update(fighter_data)
                frame_data.append(base)

            timeline.append(frame_data)

    return timeline


def cache_dataset(root_dir, char_subset=[]):
    # Maps video->fighter->frame_num to (image, label). This keeps all the frames side by side.
    video_to_sample = {}
    # Maps fighter->move to (video, frame). Use this to get a initial start, then switch over
    # to video_to_sample to get the surrounding
    move_to_frames = {}

    # Iterate through all subdirectories in root_dir
    for video_dir in os.scandir(root_dir):
        if video_dir.is_dir():
            video_name = video_dir.name
            video_to_sample[video_name] = {}

            # Iterate through all subdirectories in video_dir
            for fighter_dir in os.scandir(video_dir.path):
                if fighter_dir.is_dir():
                    # The directory structure is <fighter_id>_<fighter_name>. Remove the fighter_id.
                    fighter_name = " ".join(fighter_dir.name.split("_")[1:]).title()

                    # If char_subset is not empty, check if fighter_name is in char_subset
                    if char_subset and fighter_name not in char_subset:
                        continue

                    video_to_sample[video_name][fighter_name] = []

                    image_dir = os.path.join(fighter_dir.path, "images")
                    label_dir = os.path.join(fighter_dir.path, "labels")

                    # Create a list of all .jpg files in the images directory
                    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))

                    # Create a list of all .txt files in the labels directory
                    label_files = sorted(glob.glob(os.path.join(label_dir, "*.txt")))

                    # Assuming there is a 1-to-1 correspondence between image_files and label_files
                    video_to_sample[video_name][fighter_name].extend(
                        list(zip(image_files, label_files))
                    )

                    for frame_num, label_file in enumerate(label_files):
                        with open(label_file) as f:
                            action = f.read()

                        if fighter_name not in move_to_frames:
                            move_to_frames[fighter_name] = {}

                        if action not in move_to_frames[fighter_name]:
                            move_to_frames[fighter_name][action] = []

                        move_to_frames[fighter_name][action].append((video_name, frame_num))

            # If the video doesn't have the characters we care about, then ignore it.
            if not video_to_sample[video_name]:
                del video_to_sample[video_name]

    return video_to_sample, move_to_frames


def load_ground_truth_pairings_from_file(file_path):
    pairings = []
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip the header

        index = 0
        for row in reader:
            index += 1
            if any(cell.startswith("#") for cell in row):
                # print(f"Skipping row {index}")
                continue  # Skip this row, go to next row

            # process the row
            # print("row: ", row)
            pairings.append(tuple((row[0], row[1], row[2], int(row[3]))))

    return pairings


def update_fighters_from_timeline(frame_number: int, ground_truth, fighters):
    # Initialization
    # Sort the ground truth by fighter_id. Note that fighter_id can be all over the place. For
    # example, I've seen p1 = 0, p2 = 4.
    ground_truth = sorted(ground_truth, key=lambda x: x["fighter_id"])
    if not fighters or frame_number == 0:
        for json_data in ground_truth:
            fighter = Fighter(frame_num=frame_number, data=json_data)
            fighters.append(fighter)

    # Update
    else:
        for i, json_data in enumerate(ground_truth):
            fighters[i].update(frame_number, json_data)

    return fighters


def load_ground_truth_from_path(
    label_path: str, validate: bool = True, log_offset: int = 0, max_lines=0
):
    # ground_truth is an array where each item has the number of players on screen.
    # ground_truth[-1]:
    # [
    #   {'camera_fov': 17.0, 'camera_position': {...}, 'camera_target_position': {...}, ....},
    #   {'camera_fov': 17.0, 'camera_position': {...}, 'camera_target_position': {...}, ....}
    # ]
    ground_truth = []

    prev_num_frames_left = -1
    index = 0
    offset_count = 0

    # Duplicate initial state. THIS DOES NOT WORK.
    if log_offset < 0:
        with open(label_path, "r") as f:
            line1 = json.loads(f.readline())
            line2 = json.loads(f.readline())
            print(f"Duplicating {line1} and {line2}")
            ground_truth = [[line1, line2]] * abs(log_offset)
            print(f"Adding {abs(log_offset)} starter lines")
            index += 2 * abs(log_offset)
            log_offset = 0

    with open(label_path, "r") as f:
        for line in f:
            if max_lines and index > max_lines:
                break

            # considering each line is a half of a frame, I need to double it.
            if offset_count < (2 * log_offset):
                offset_count += 1
                continue

            json_data = json.loads(line)
            # This assumes only one fighter.  For each frame, there are two lines in the log
            # since there is one line per fighter.
            frame_number = index // 2
            if frame_number >= len(ground_truth):
                ground_truth.append([])

            # Check to make sure the log didn't miss any frames. If it did, repeat the latest one N
            # times.
            diff = prev_num_frames_left - json_data["num_frames_left"]
            if prev_num_frames_left > 0 and diff > 1:
                # print(f"Got missing frame_data for line {index} with diff {diff}")
                # print(f"Previously had {len(ground_truth)} gt, index at {index}")
                repeated_logs = [ground_truth[-1]] * (diff - 1)
                ground_truth += repeated_logs
                index += (diff - 1) * 2
                # print(f"Now have {len(ground_truth)} gt, index at {index}")

            ground_truth[frame_number].append(json_data)
            index += 1

            prev_num_frames_left = json_data["num_frames_left"]

    # Manually overwrite fighter_id to make it 0 and 1. Hack.
    for i, frame_data in enumerate(ground_truth):
        frame_data = sorted(frame_data, key=lambda x: x["fighter_id"])
        for j, fighter_data in enumerate(frame_data):
            fighter_data["fighter_id"] = j
        ground_truth[i] = frame_data

    if validate:
        # assert (
        #     index == (len(ground_truth) * 2) - 1
        # ), f"index should be 2x - 1 length of ground truth {index} != {len(ground_truth) * 2}"
        for i in range(len(ground_truth)):
            gt = ground_truth[i]
            assert len(gt) == 2, (
                "there should be the ground truth for 2 players for every frame, found "
                + f"{len(gt)} for frame #{i}"
            )
    return ground_truth


class GTVideo:
    def __init__(self, video_path, label_path):
        self.video_path = video_path
        self.label_path = label_path
        self.fps = None
        self.ground_truth = []

    def get_fps(self):
        if self.fps is None:
            video = cv2.VideoCapture(self.video_path)
            self.fps = video.get(cv2.CAP_PROP_FPS)
            video.release()
        return self.fps

    def get_frame(self, frame_number):
        video = cv2.VideoCapture(self.video_path)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = video.read()
        video.release()
        return frame

    def get_frames(self, start_frame, num_frames):
        frames = []
        for i in range(start_frame, start_frame + num_frames):
            frames.append(self.get_frame(i))
        return frames

    def load_ground_truth(self):
        self.ground_truth = load_ground_truth_from_path(self.label_path)

    def get_ground_truth(self, start_frame, num_frames):
        if not self.ground_truth:
            self.load_ground_truth()

        return [self.ground_truth[i] for i in range(start_frame, num_frames)]

    def get_frames_and_labels(self, start_frame, num_frames):
        frames = self.get_frames(start_frame, num_frames)
        ground_truth = self.get_ground_truth(start_frame, num_frames)
        return frames, ground_truth
