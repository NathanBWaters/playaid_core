"""
DEPRECATED - this is old CV work, not using ground truth data.
"""

import os
import click
from collections import defaultdict
from datetime import datetime
import yaml
import random
import cv2
from addict import Dict
import torch
import shutil
import subprocess
import re
import numpy as np
import glob
import imutils
import json
from PIL import ImageOps, Image
import easyocr
from paddleocr import PaddleOCR

from playaid.anim_ontology import MOVE_TO_ADVANTAGE_STATE, MOVE_TO_CLASS_ID
from playaid.dataset_utils import action_sample_from_frame_middle_out
from playaid.timeline import (
    load_ground_truth_from_path,
    load_ground_truth_pairings_from_file,
    update_fighters_from_timeline,
)
from playaid.models.cnn_action_detector import CNNActionDetector
from playaid.fighter import YoloCrop
from playaid import constants


def extract_number_from_filename(filename):
    """
    Files are in the following naming convention:
    ai_cache/ult_videos/tweek-mkleo-clip/crops/Byleth/tweek-mkleo-game-1_1.jpg
    ai_cache/ult_videos/tweek-mkleo-clip/crops/Byleth/tweek-mkleo-game-1_2.jpg
    or
    ai_cache/ult_videos/tweek-mkleo-clip/labels/Byleth/tweek-mkleo-game-1_1.txt
    ai_cache/ult_videos/tweek-mkleo-clip/labels/Byleth/tweek-mkleo-game-1_2.txt
    """
    match = re.search(r"(\d+)(?=\.\w+$)", filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError("Cannot get number from filename {filename}")


def read_fighter_yolo_crop(label_path, fighter):
    with open(label_path) as file:
        for line in file.readlines():
            assert (
                len(line.split(" ")) == 6
            ), f"Too much data for line: {line} in label {label_path}"

            class_id, center_x, center_y, width, height, confidence = line.split(" ")
            if int(class_id) == constants.CHAR_LIST.index(fighter):
                return YoloCrop(
                    float(center_x),
                    float(center_y),
                    float(width),
                    float(height),
                    confidence=float(confidence),
                    class_id=int(class_id),
                )

    return None


def read_yolo_crops(label_path):
    crops = []
    with open(label_path) as file:
        for line in file.readlines():
            assert (
                len(line.split(" ")) == 6
            ), f"Too much data for line: {line} in label {label_path}"

            class_id, center_x, center_y, width, height, confidence = line.split(" ")
            crops.append(
                YoloCrop(
                    float(center_x),
                    float(center_y),
                    float(width),
                    float(height),
                    confidence=float(confidence),
                    class_id=int(class_id),
                )
            )

    return crops


def extract_numbers(text):
    numbers = re.findall(r"\d+", text)
    return "".join(numbers)


def check_newline(file_path):
    with open(file_path, "rb") as f:
        f.seek(-1, 2)  # Go to the 2nd last byte.
        last_byte = f.read(1)
    return last_byte == b"\n"


def damage_crop_to_percent(damage_crop, paddle_ocr):
    """
    Given a crop of just the damage, return the percent as a float
    """
    # results = ocr_reader.readtext(damage_crop)
    damage_crop = imutils.resize(damage_crop, width=256)
    results = paddle_ocr(damage_crop)
    boxes, detected_text, extra = results

    if len(detected_text) == 2:
        whole_number, decimal = (
            (detected_text[0][0], detected_text[1][0])
            if boxes[0][0][0] < boxes[1][0][0]
            else (detected_text[1][0], detected_text[0][0])
        )

        if whole_number != "" and decimal != "":
            return True, (
                float(extract_numbers(whole_number) + "." + extract_numbers(decimal)),
                whole_number + "." + decimal,
                detected_text[0][1],
                (boxes, detected_text, extra),
            )

    return False, (-1, "_".join([r[0] for r in detected_text]), 0.0, results)


class AIRunner:
    """
    Runs e2e tracking and action recognition.
    """

    def __init__(
        self,
        input_video_path: str = os.path.join(
            constants.ULT_DATASET_DIR, "ult_videos/tweek-mkleo-clip.mp4"
        ),
        debug: bool = False,
        **dataset_args,
    ):
        self.input_video_path = input_video_path
        self.src_folder, self.file_name = os.path.split(self.input_video_path)
        # Split the filename into root and extension
        self.video_name, _ = os.path.splitext(self.file_name)
        self.input_video = cv2.VideoCapture(input_video_path)

        # Extract parent folder name
        parent_folder = os.path.basename(self.src_folder)
        self.exp_name = os.path.join(parent_folder, self.video_name)
        self.yolo_output_dir = os.path.join(constants.AI_CACHE, self.exp_name)
        self.ai_output_file = os.path.join(self.yolo_output_dir, "ai_output.yaml")
        self.crops_dir = os.path.join(self.yolo_output_dir, "crops")
        self.labels_dir = os.path.join(self.yolo_output_dir, "labels")
        self.dataset_args = dataset_args

        self.model = CNNActionDetector.load_from_checkpoint(
            os.path.join(constants.SAVED_ACTION_MODELS, "four-chars-aug-4.ckpt"),
            actions=list(MOVE_TO_CLASS_ID.keys()),
        )
        self.model.eval()
        res, self.ai_output_data = self.load_ai_output()
        print(f"Has previously created ai_output: ${res}")

        self.debug = debug
        now = datetime.now()
        date_time_str = now.strftime("%Y-%m-%d-%H:%M:%S")
        self.debug_path = os.path.join(self.yolo_output_dir, f"debug-{date_time_str}")
        if self.debug and not os.path.exists(self.debug_path):
            os.makedirs(self.debug_path)

        self.run_detection_setup()

    def run_detection_setup(self):
        self.run_yolo()
        self.fighters = [
            fighter
            for fighter in os.listdir(self.crops_dir)
            if os.path.isdir(os.path.join(self.crops_dir, fighter))
        ]

        self.clean_yolo_crops()

    def run_yolo(self):
        if os.path.exists(self.crops_dir):
            print(f"Yolo has already been run on {self.input_video_path}")
            return

        # Create ffmpeg command
        command = [
            "python",
            os.path.join(constants.YOLO_DIR, "detect.py"),
            "--weights",
            os.path.join(constants.SAVED_YOLO_MODELS, "byleth-diddy-pikachu-joker-july-31-2023.pt"),
            "--source",
            self.input_video_path,
            "--project",
            constants.AI_CACHE,
            "--name",
            self.exp_name,
            "--max-det",
            "2",
            "--save-crop",
            "--save-txt",
            "--save-conf",
            "--exist-ok",
            # HACK, REMOVE THIS LATER
            "--classes",
            "2",
            "3",
        ]

        print(f"Running command {' '.join(command)}")
        # Execute command
        subprocess.run(command, check=True)

        return self.exp_name

    def clean_yolo_crops(self):
        """
        We write files to AI_CACHE.
        If YOLO misses the character, then we'll interpolate the crop here.
        """
        # if Yolo thinks there's more than two characters, remove the ones that have the least
        # number of labels
        num_fighters = len(
            [
                fighter
                for fighter in os.listdir(self.crops_dir)
                if os.path.isdir(os.path.join(self.crops_dir, fighter))
            ]
        )
        if num_fighters != 2:
            print("Detected too many characters")
            exit()

        last_frame_path = self.get_label_paths()[-1]
        self.max_frames = extract_number_from_filename(last_frame_path)

        # If the same character is detected twice for the same frame, then Yolo will make both
        # crops but the second one will be another number attached at the end. This can can cause
        # crops with very large frame numbers and they should be removed.
        for fighter in self.fighters:
            crop_paths = self.get_crop_paths(fighter)
            for crop_path in reversed(crop_paths):
                if extract_number_from_filename(crop_path) <= self.max_frames:
                    break

                print(f"Removing too large of number {crop_path}")
                os.unlink(crop_path)

        # Make sure we at least have a file for each crop
        for i in range(1, self.max_frames):
            path = self.get_label_path(i)
            if not os.path.exists(path):
                print(f"Creating empty crop: {path}")
                with open(path, "w"):
                    pass

        for fighter in self.fighters:
            self.clean_yolo_crops_for_fighter(fighter)

        # Handle missing crops at the end. This is a dumb approach that simply copies over the last
        # frame for a fighter.
        fighter_to_max_frames = {
            fighter: extract_number_from_filename(self.get_crop_paths(fighter)[-1])
            for fighter in self.fighters
        }
        max_frames = max(list(fighter_to_max_frames.values()))
        for fighter, last_frame_num in fighter_to_max_frames.items():
            num_remaining_frames = max_frames - last_frame_num
            if not num_remaining_frames:
                continue

            last_frame_path = self.get_crop_paths(fighter)[-1]
            last_frame = cv2.imread(last_frame_path)
            print(
                f"For {fighter} duplicating last frame {last_frame_num} {num_remaining_frames} times"
            )
            for i in range(last_frame_num, last_frame_num + num_remaining_frames):
                end_path = self.get_crop_path(fighter, i)
                cv2.imwrite(end_path, last_frame)

    def get_label_path(self, frame_num):
        return os.path.join(self.labels_dir, f"{self.video_name}_{frame_num}.txt")

    def get_crop_path(self, fighter, frame_num):
        return os.path.join(self.crops_dir, fighter, f"{self.video_name}_{frame_num}.jpg")

    def get_label_paths(self):
        label_paths = glob.glob(os.path.join(self.labels_dir, "*.txt"))
        return sorted(label_paths, key=extract_number_from_filename)

    def get_crop_paths(self, fighter):
        fighter_dir = os.path.join(self.crops_dir, fighter)
        crop_paths = glob.glob(os.path.join(fighter_dir, "*.jpg"))
        return sorted(crop_paths, key=extract_number_from_filename)

    def clean_yolo_crops_for_fighter(self, fighter):
        crop_paths = self.get_crop_paths(fighter)
        label_paths = self.get_label_paths()

        # Make sure there are no duplicates. If there's a duplicate, choose the nearest
        # TODO: this is done unecessarilly twice.
        # TODO: this doesn't work on the first frame.
        previous_class_id_to_crop = {}
        for i, label_path in enumerate(label_paths):
            class_id_to_crop = defaultdict(list)
            yolo_crops = read_yolo_crops(label_path)
            for crop in yolo_crops:
                class_id_to_crop[crop.class_id].append(crop)

            found_duplicate = False
            for class_id, crops in class_id_to_crop.items():
                if len(crops) > 1 and class_id in previous_class_id_to_crop:
                    found_duplicate = True
                    min_distance, nearest_crop = 10000, None

                    for crop in crops:
                        distance = abs(
                            crop.center_x - previous_class_id_to_crop[class_id].center_x
                        ) + abs(crop.center_y - previous_class_id_to_crop[class_id].center_y)
                        if distance < min_distance:
                            min_distance = distance
                            nearest_crop = crop

                    class_id_to_crop[class_id] = [nearest_crop]

            # Write the cleaned output
            new_yolo_strings = []
            for class_id, crops in class_id_to_crop.items():
                assert len(crops) == 1, "We should have cleaned out the duplicates at this point"
                new_yolo_strings.append(str(crops[0]))
                previous_class_id_to_crop[class_id] = crops[0]

            if not found_duplicate:
                continue

            old_content = "\n".join([str(crop) for crop in yolo_crops])
            new_content = "\n".join(new_yolo_strings) + "\n"
            print(
                f"Re-writing {label_path} from:\n{old_content}\nto:\n{new_content}\n"
                + str(previous_class_id_to_crop)
            )
            with open(label_path, "w") as f:
                f.write(new_content)

            with open(label_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    assert len(line.split(" ")) == 6, "have a non-6 split line..."

        # Handle missing crops before first detected frame
        # TODO
        latest_seen_frame = extract_number_from_filename(label_paths[0])

        # Handle missing crops interpolation
        for i, crop_path in enumerate(crop_paths):
            current_frame = extract_number_from_filename(crop_path)
            if current_frame - latest_seen_frame > 1:
                # We now have a need to interpolate
                print(f"Missing frames {latest_seen_frame + 1}-{current_frame - 1} for {fighter}")
                latest_label = self.get_label_path(latest_seen_frame)
                current_label = self.get_label_path(current_frame)
                start_yolo_crop = read_fighter_yolo_crop(latest_label, fighter)
                assert (
                    start_yolo_crop
                ), f"missing start_yolo_crop {latest_label} for {fighter} | {crop_path}"
                end_yolo_crop = read_fighter_yolo_crop(current_label, fighter)
                assert (
                    end_yolo_crop
                ), f"missing end_yolo_crop {current_label} for {fighter} | {crop_path}"

                for j in range(latest_seen_frame + 1, current_frame):
                    # There's an odd scenario where we don't have a crop for a frame, we can't
                    # read the frame back for the video, and we've already written the
                    # interpolation. If exists, just continue.
                    if read_fighter_yolo_crop(self.get_label_path(j), fighter):
                        print("Already have the intermediate crop")
                        continue

                    interp_percent = (current_frame - j) / (current_frame - latest_seen_frame)
                    interp_crop = start_yolo_crop.interp(end_yolo_crop, percent=interp_percent)

                    output_label_path = self.get_label_path(j)
                    # has_newline = check_newline(output_label_path)
                    with open(output_label_path, "a") as f:
                        print(f"Adding {str(interp_crop)} to {output_label_path}")
                        f.write(str(interp_crop) + "\n")

                    with open(output_label_path, "r") as f:
                        lines = f.readlines()
                        # assert len(lines) == 2, "only 2 lines allowed"
                        for line in lines:
                            assert len(line.split(" ")) == 6, "have a non-6 split line..."

                    self.input_video.set(cv2.CAP_PROP_POS_FRAMES, j)
                    res, input_frame = self.input_video.read()
                    if not res:
                        print(
                            f"Failed to read from frame {j} during interpolation, just going to "
                            + "copy over previous frame"
                        )

                        shutil.copy(
                            self.get_crop_path(fighter, j - 1), self.get_crop_path(fighter, j)
                        )
                        continue

                    output_dimension = 128
                    res, crop = interp_crop.square_crop(input_frame, output_dimension, padding=30)
                    assert res, f"Failed to get square crop from frame {j}"
                    output_crop_path = self.get_crop_path(fighter, j)
                    print(f"Adding crop {output_crop_path}")
                    cv2.imwrite(output_crop_path, crop)

            latest_seen_frame = current_frame

    def get_action_recognition_input_for_frame(self, frame: int, fighter: str):
        """
        Given a frame, gets the input for the action recognition model
        """
        frame_nums = action_sample_from_frame_middle_out(
            frame,
            num_frames_per_sample=self.dataset_args.get("num_frames_per_sample", 7),
            frame_delta=(
                random.choice(self.dataset_args["frame_delta"])
                if "frame_delta" in self.dataset_args
                else 3
            ),
            max_frames=self.max_frames,
            min_frame=1,
        )

        frames = []
        for frame_num in frame_nums:
            output_size = 128
            frame_path = self.get_crop_path(fighter, frame_num)
            frame = cv2.imread(frame_path)
            assert frame is not None, f"Failed to get frame {frame_path}"
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resizes but maintains aspect ratio.
            frame = imutils.resize(frame, width=output_size, height=output_size)

            if frame.shape[0] != output_size or frame.shape[1] != output_size:
                # Pad it to make it always output_size x output_size with a black letterbox.
                frame = np.array(
                    ImageOps.pad(Image.fromarray(frame), (output_size, output_size), color="black")
                )

            assert frame.shape == (128, 128, 3), f"Bad shape {frame.shape}@{frame_path}"
            frames.append(frame)

        input_frames = torch.tensor(np.array(frames))
        input_frames = input_frames.permute(0, 3, 1, 2).unsqueeze(0)
        input_frames = input_frames.float() / 255.0
        return input_frames, frames

    def action_recognition(self, frame_num: int, fighter: str):
        """
        Runs action recognition for a given frame and fighter.
        """
        input_frames, frames = self.get_action_recognition_input_for_frame(frame_num, fighter)

        predictions = self.model(input_frames)

        predicted_action_id = int(torch.argmax(predictions))
        predicted_action = self.model.actions[predicted_action_id]
        probabilities = torch.exp(predictions)
        confidence = float(probabilities[0][predicted_action_id]) * 100.0
        crop_path = self.get_label_path(frame_num)
        crop = read_fighter_yolo_crop(crop_path, fighter)
        return (
            input_frames,
            constants.CHAR_LIST.index(fighter),  # TODO
            torch.tensor(predicted_action_id),
            {
                "char": fighter,
                "predicted_action": predicted_action,
                "confidence": confidence,
                "crop": crop,
                "frames": [np.array(f) for f in frames],
            },
        )

    def run_action_recognition(self, overwrite=False):
        """
        For each frame:
        - get the necessary surrounding frames
        - read in the images
        - resize those images
        - pass the images into the action recognition model
        - write the original crop and predicted action to a json file
        """
        for fighter in self.fighters:
            if not overwrite and self.ai_output_data[fighter][0].action:
                print(f"Already performed action recognition for {fighter}")
                continue

            for frame_num in range(1, self.max_frames):
                input_frames, class_id, predicted_action_id, data = self.action_recognition(
                    frame_num, fighter
                )
                predicted_action = data["predicted_action"]
                confidence = data["confidence"]
                crop = data["crop"]
                print(f"{fighter}@{frame_num} = {predicted_action} @ {confidence:.2f}")

                # Yolo is 1 indexed, switch to 0 indexed.
                frame_data = self.ai_output_data[fighter][frame_num - 1]
                frame_data.crop = str(crop)
                frame_data.action = predicted_action
                frame_data.predicted_action_confidence = confidence

    def determine_player_id_to_fighter(self):
        """
        TODO
        """
        self.player_id_to_fighter = {}
        # starting_position_path = self.get_label_paths()[0]
        # yolo1, yolo2 = read_yolo_crops(starting_position_path)
        # player_1_class, player_2_class = (
        #     (yolo1.class_id, yolo2.class_id)
        #     if yolo1.center_x < yolo2.center_x
        #     else (yolo2.class_id, yolo1.class_id)
        # )
        self.player_id_to_fighter[0] = "Pikachu"
        self.player_id_to_fighter[1] = "Joker"

    def run_damage_detection(self):
        """
        For each frame:
        - get the necessary surrounding frames
        - read in the images
        - resize those images
        - pass the images into the action recognition model
        - write the original crop and predicted action to a json file
        """
        self.determine_player_id_to_fighter()

        self.ocr = PaddleOCR(
            use_angle_cls=True, lang="en", show_log=False
        )  # need to run only once to download and load model into memory

        num_confident = 0
        player_id_damage_pairs = [
            (0, 402 / 1280),
            (1, 898 / 1280),
        ]
        for i in range(self.max_frames):
            self.input_video.set(cv2.CAP_PROP_POS_FRAMES, i)
            res, input_frame = self.input_video.read()
            if not res:
                print("Frame {i} could not be read for damage output")
                continue
            for player_id, damage_x in player_id_damage_pairs:
                damage_img = YoloCrop(
                    center_x=damage_x,
                    center_y=637 / 720,
                    crop_width=133 / 1280,
                    crop_height=60 / 720,
                ).crop_img(input_frame)

                res, (damage, original_string, confidence, results) = damage_crop_to_percent(
                    damage_img, self.ocr
                )
                num_confident += res

                if self.debug:
                    cv2.imwrite(
                        os.path.join(
                            self.debug_path,
                            f"{i}_p{player_id}_{'_' if res else 'FAIL_'}{damage}_{original_string}.jpg",
                        ),
                        damage_img,
                    )

                    print(f"Percent confident: {(num_confident / ((i + 1) * 2)):.2f}")

                fighter = self.player_id_to_fighter[player_id]
                self.ai_output_data[fighter][i].damage = damage

        print(f"#{i} Percent confident: {(num_confident / (self.max_frames * 2)):.2f}")

    def load_ai_output(self):
        """
        Loads the existing AI output if it exists.
        """
        if not os.path.exists(self.ai_output_file):
            return False, Dict()

        with open(self.ai_output_file, "r") as f:
            try:
                data = yaml.safe_load(f)
                return True, Dict(data)
            except Exception:
                return False, Dict()

    def write_output(self):
        with open(self.ai_output_file, "w") as f:
            yaml.dump(self.ai_output_data.to_dict(), f)


@click.command()
@click.option("--video", "-v", help="Path to video")
def ai_runner(video):
    """Entrypoint to AIRunner"""
    runner = AIRunner(input_video_path=video, debug=True)
    runner.load_ai_output()
    runner.run_action_recognition()
    runner.write_output()
    runner.run_damage_detection()
    runner.write_output()

    print("🎉 COMPLETED 🎉")


if __name__ == "__main__":
    ai_runner()
