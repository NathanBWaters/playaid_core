"""
Takes in the cleaned animation data from ult_dataloader and uses the stage background to create
synthetic smash videos. Outputs data in the expected AVA dataset format for FB SlowFast action
recognition models to train on.

ult_dataset/
└── synth_char_action_recognition/
    ├── annotations/
    │   ├── train.csv
    │   ├── validation.csv
    │   ├── test.csv
    │   └── class_ids.json
    ├── frames/
    │   ├── train.txt
    │   ├── validation.txt
    │   ├── video_name_1/
    │   │   ├── video_name_1_000001.jpg
    │   │   ├── video_name_1_000002.jpg
    │   │   └── etc...  
    │   ├── video_name_2/
    │   └── etc.../
    └── videos/

The annotations/{train/validation/test}.csv files will be in the format (but won't actually have
a header for some reason)
video_name, frame_sec (int), x1,y1,x2,y2 (with a range of 0 to 1), class_id, ??
_-Z6wFjXtGQ,0902,0.063,0.049,0.524,0.996,12,0
_-Z6wFjXtGQ,0902,0.063,0.049,0.524,0.996,74,0
_-Z6wFjXtGQ,0902,0.063,0.049,0.524,0.996,80,0
_-Z6wFjXtGQ,0902,0.392,0.253,0.916,0.994,12,1
_-Z6wFjXtGQ,0902,0.392,0.253,0.916,0.994,17,1
_-Z6wFjXtGQ,0902,0.392,0.253,0.916,0.994,74,1
_-Z6wFjXtGQ,0903,0.032,0.048,0.494,0.980,12,0

The frames/{train/validation}.txt (which I'm not sure why this is needed) files will be in the
format (and will actually have a header):
original_vido_id video_id frame_id path labels
-5KQ66BBWC4 0 0 -5KQ66BBWC4/-5KQ66BBWC4_000001.jpg ""
-5KQ66BBWC4 0 1 -5KQ66BBWC4/-5KQ66BBWC4_000002.jpg ""
-5KQ66BBWC4 0 2 -5KQ66BBWC4/-5KQ66BBWC4_000003.jpg ""


The class_ids.json will be in the format:{
    "class_name_0": 0,
    "class_name_1": 1,
    etc..
}
"""
from glob import glob
import cv2
import numpy as np
import os
from pathlib import Path
from PIL import Image
import random
import imutils
import shutil

from playaid.constants import (
    CHAR_LIST,
    SYNTH_ACTION_RECOGNITON_DIR,
    SYNTH_ACTION_RECOGNITON_FRAMES_DIR,
    SYNTH_ACTION_RECOGNITON_ANNOTATIONS_DIR,
    ULT_DATASET_CLEAN_CHAR_DIR,
)
from playaid.anim_ontology import MOVE_TO_CLASS_ID
from playaid.dataset_utils import get_stage_paths, get_character_animations_dict


class SynthCharacter:
    """
    Stores data about a synthetic character.
    """

    def __init__(self, fighter_name: str, x: int, y: int):
        """
        @param fighter_name: name of character like "diddy_kong" or "byleth"
        @param x: x position on stage in pixels
        @param y: y position on stage in pixels
        """
        self.fighter_name = fighter_name
        self.center_x = x
        self.center_y = y

        self.moves = [
            x[0].split("/")[-1]
            for x in os.walk(
                os.path.join(ULT_DATASET_CLEAN_CHAR_DIR, self.fighter_name)
            )
            if x[0].split("/")[-1] != "Undefined"
            and x[0].split("/")[-1] != self.fighter_name
        ]

        self.scale = random.choice([0.2, 0.25, 0.3])

        self.animation_paths = []

    def label(self):
        return MOVE_TO_CLASS_ID[self.move]

    def load_animations(self):
        self.move = random.choice(self.moves)

        animation_dir = animation_pattern = os.path.join(
            ULT_DATASET_CLEAN_CHAR_DIR, self.fighter_name, self.move
        )
        body_types = set(())
        raw_animation_names = set(())
        cam_directions = set(())

        for abs_file_path in glob(os.path.join(animation_dir, "*.png")):
            file_name = Path(abs_file_path).stem
            (
                _,
                body_type,
                raw_anim_name,
                frame,
                cam_direction,
                frame_num,
            ) = file_name.split("_")
            body_types.add(body_type)
            raw_animation_names.add(raw_anim_name)
            cam_directions.add(cam_direction)

        animation_grouping = (
            f"{self.fighter_name}_{random.choice(list(body_types))}_"
            + f"{random.choice(list(raw_animation_names))}_frame_{random.choice([-90, 90])}_*.png"
        )

        animation_pattern = os.path.join(
            ULT_DATASET_CLEAN_CHAR_DIR,
            self.fighter_name,
            self.move,
            animation_grouping,
        )
        self.animation_paths = glob(animation_pattern)

        self.animation_paths = sorted(
            self.animation_paths, key=lambda x: int(Path(x).stem.split("_")[-1])
        )

        if not self.animation_paths:
            print("hmmmm")

    def bbox_yolo(self):
        return (
            self.center_x,
            self.center_y,
            self.char_image.width,
            self.char_image.height,
        )

    def bbox_yolo_norm(self, width, height):
        bbox = self.bbox_yolo()
        return (
            bbox[0] / width,
            bbox[1] / height,
            bbox[2] / width,
            bbox[3] / height,
        )

    def bbox_corners(self):
        x_center, y_center, width, height = self.bbox_yolo()
        top = y_center - (height / 2)
        bottom = y_center + (height / 2)
        left = x_center - (width / 2)
        right = x_center + (width / 2)
        return (
            (int(left), int(top)),
            (int(right), int(top)),
            (int(left), int(bottom)),
            (int(right), int(bottom)),
        )

    def tick(self):
        """
        Moves the animation forward one frame.
        Returns the frame path.
        """
        if not self.animation_paths:
            self.load_animations()

        char_image = Image.open(self.animation_paths.pop(0))

        # Resize the character to a more typical size.
        self.char_image = char_image.resize(
            (int(char_image.width * self.scale), int(char_image.height * self.scale))
        )


class SynthVideoGenerator:
    """
    Generates synthetic smash videos and their corresponding labels.
    """

    def __init__(
        self,
        num_videos_per_split=None,
        overwrite=False,
        video_length=60,
        debug=False,
    ):
        """ """
        if not num_videos_per_split:
            num_videos_per_split = {"train": 1000, "validation": 32, "test": 32}

        self.num_videos_per_split = num_videos_per_split

        self.stage_paths = get_stage_paths()
        self.video_length = video_length
        self.width = 1280
        self.height = 960
        self.char_animations = get_character_animations_dict()
        self.debug = debug
        self.frame_num = 0
        self.overwrite = overwrite
        self.video_id = 0

        # Not actually used.
        self.excluded_file_path = os.path.join(
            SYNTH_ACTION_RECOGNITON_ANNOTATIONS_DIR, "excluded.csv"
        )
        self.label_map_file = os.path.join(
            SYNTH_ACTION_RECOGNITON_ANNOTATIONS_DIR, "label_map_file.pbtxt"
        )

        if self.overwrite and os.path.exists(SYNTH_ACTION_RECOGNITON_DIR):
            shutil.rmtree(SYNTH_ACTION_RECOGNITON_DIR)

    def generate(self):
        """ """
        for split in list(self.num_videos_per_split.keys()):
            print(f" --- On split {split} --- ")
            for i in range(1, self.num_videos_per_split[split] + 1):
                # absolute path to split
                self.annotation_csv_file_path = os.path.join(
                    SYNTH_ACTION_RECOGNITON_ANNOTATIONS_DIR, split + ".csv"
                )
                # I don't understand why this is necessary but it's how SlowFast is split up.
                self.annotation_txt_file_path = os.path.join(
                    SYNTH_ACTION_RECOGNITON_FRAMES_DIR, split + ".txt"
                )
 
                print(f"Creating video #{self.video_id}")
                self.gen_frames(self.video_id)

                self.video_id += 1

        self.on_complete()

    def on_complete(self):
        """"""
        # Just need to make an empty excluded.csv
        with open(self.excluded_file_path, "w") as excluded_file:
            excluded_file.close()

        # Create a json converting the labels into names.
        with open(self.label_map_file, "w") as label_map_file:
            for move, label_id in MOVE_TO_CLASS_ID.items():
                label_map_file.write("item {\n")
                label_map_file.write(f'  name: "{move}"\n')
                label_map_file.write(f"  id: {label_id}\n")
                label_map_file.write("}\n")

    def init_characters(self, num_characters):
        """
        Initializes characters for the video.
        """
        characters = []
        for i in range(num_characters):
            center_x = int(random.gauss(self.width / 2, self.width / 6))
            center_y = int(random.gauss(self.height / 2, self.height / 6))
            # Just put it in the middle if the image falls outside the expected range.
            if center_x < 0 or center_x > self.width:
                center_x = self.width / 2
            if center_y < 0 or center_y > self.height:
                center_y = self.height / 2

            fighter_name = random.choice(CHAR_LIST)
            character = SynthCharacter(fighter_name, center_x, center_y)
            characters.append(character)

        return characters

    def write_annotation(
        self, video_name: str, characters, frame_num: int, output_file_name: str
    ):
        """ """
        if not os.path.exists(SYNTH_ACTION_RECOGNITON_ANNOTATIONS_DIR):
            os.makedirs(SYNTH_ACTION_RECOGNITON_ANNOTATIONS_DIR)

        with open(self.annotation_csv_file_path, "a") as annotation_file:
            for character in characters:
                yolo = character.bbox_yolo_norm(self.width, self.height)
                annotation_file.write(
                    f"{video_name}, {frame_num}, {yolo[0]}, {yolo[1]}, {yolo[2]}, {yolo[3]}, "
                    + f"{character.label()}, 1.0\n"
                )

        if not os.path.exists(self.annotation_txt_file_path):
            with open(self.annotation_txt_file_path, "a") as annotation_file:
                # Yes, "original_vido_id" is mispelled on purpose to follow AVA dataset format.
                annotation_file.write(
                    "original_vido_id video_id frame_id path labels\n"
                )

        with open(self.annotation_txt_file_path, "a") as annotation_file:
            for character in characters:
                yolo = character.bbox_yolo_norm(self.width, self.height)
                annotation_file.write(
                    f"{video_name} {self.video_id} {frame_num} "
                    + f'{os.path.join(video_name, output_file_name)} ""\n'
                )

    def gen_frames(self, video_index):
        """"""
        stage_path = random.choice(self.stage_paths)

        characters = self.init_characters(2)

        video_name = f"video_{video_index}"
        output_dir = os.path.join(SYNTH_ACTION_RECOGNITON_FRAMES_DIR, video_name)

        if self.overwrite and os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for frame_num in range(self.video_length):
            # Start with a clean slate each time.
            stage = cv2.imread(stage_path)
            # Keep the aspect ratio the same but crop the height.
            stage = cv2.cvtColor(stage, cv2.COLOR_BGR2RGB)
            stage = imutils.resize(stage, width=self.width)
            stage = Image.fromarray(stage)

            output_file_name = f"{video_name}_{str(frame_num).zfill(6)}.jpg"
            output_path = os.path.join(output_dir, output_file_name)

            for character in characters:
                character.tick()

                stage.paste(
                    character.char_image,
                    # paste just wants upper left coordinate.
                    (
                        int(character.center_x - (character.char_image.width / 2)),
                        int(character.center_y - (character.char_image.height / 2)),
                    ),
                    character.char_image,
                )

            stage = cv2.cvtColor(np.asarray(stage), cv2.COLOR_BGR2RGB)
            cv2.imwrite(output_path, stage)

            if self.debug:
                print(f"Wrote {output_path}")

            self.write_annotation(video_name, characters, frame_num, output_file_name)


if __name__ == "__main__":
    SynthVideoGenerator(
        num_videos_per_split={"train": 500, "validation": 10, "test": 10},
        video_length=120,
        overwrite=True,
        debug=True,
    ).generate()

    print("🎉 COMPLETED 🎉")
