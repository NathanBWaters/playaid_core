"""
Common utils for working with datasets in ult_datasets/
"""
import albumentations as A
import cv2
from glob import glob
import os
import random
import numpy as np
from PIL import ImageOps, Image
import math
import imutils
from pathlib import Path

from playaid.constants import ULT_STAGES_DIR, ULT_DATASET_CLEAN_CHAR_DIR
from playaid.anim_ontology import (
    ANIM_FILE_TO_ANIMATION,
    PARAM_STRING_TO_ANIMATION,
    STATUS_ENUM_TO_STRING,
)


def get_animation_type_in_dict(key: str, key_to_animation: dict) -> str:
    """
    Converts param string or animation_file to our animation animation types.
    """
    if key in key_to_animation:
        return key_to_animation[key]

    match = "Undefined"
    # Keep removing latters form the key until we have a match.  Otherwise it will be
    # "Undefined".
    for i in range(0, -1 * len(key), -1):
        if key[0:i] in key_to_animation:
            match = key_to_animation[key[0:i]]

    return match


def get_animation_type_for_param_string(param_string: str) -> str:
    """
    Converts the param_string names to our animations.
    """
    return get_animation_type_in_dict(param_string, PARAM_STRING_TO_ANIMATION)


def get_anim_for_string_and_status_kind(action_string, status_kind) -> str:
    """
    Sometimes you need the animation and the character's status
    """
    raw_action = get_animation_type_for_param_string(action_string)

    if (
        status_kind in STATUS_ENUM_TO_STRING
        and STATUS_ENUM_TO_STRING[status_kind] == "FIGHTER_STATUS_KIND_GUARD_DAMAGE"
    ):
        return "ShieldStun"

    return raw_action


def get_animation_type_for_anim_file(anim_file: str) -> str:
    """
    We only store the animation prefix.  For example, there are all of animation names for jab like:
     - c00attack100
     - c00attack100start
     - c00attack11
     - c00attack12
     - c00attack13
     - c00attack1n
     So we just store "c00attack1". This should work for the many variations.
    """
    return get_animation_type_in_dict(anim_file, ANIM_FILE_TO_ANIMATION)


def action_sample_from_frame(
    frame_num,
    num_frames_per_sample,
    frame_delta,
    valid_frames,
    fps=60,
):
    """
    @param frame_num: the last frame num in the sequence.  Work backwards.
    @param num_frames_per_sample: num of frames in the sequence.
    @param frame_delta: how many frames we skip for the sequence.
    @param valid_frames: list of valid frames
    @param fps=60:
    """
    # So if input is a triplet, zero-indexed middle frame is 1
    # So if there's five frames as input, middle frame is 2
    frame_nums = []

    # The synthetic data is 60 fps, but the ground truth is 30.

    # Get the num_frames_per_sample leading up to the current frame.
    for i in range(0, num_frames_per_sample * frame_delta, frame_delta):
        new_frame_num = frame_num - i
        if new_frame_num in valid_frames:
            frame_nums.append(new_frame_num)
        else:
            # Just use the last one we put in again.
            frame_nums.append(frame_nums[-1])

    frame_nums.reverse()
    return frame_nums


def action_sample_from_frame_middle_out(
    middle_frame, num_frames_per_sample, frame_delta, max_frames, min_frame=0, clamp=True
):
    """
    @param clamp: in production we'll want to clamp but with synthetic data we'll just grab from
    another animation if the frame nums are out of bounds.
    """
    assert num_frames_per_sample % 2 == 1, "num_frames_per_sample must be odd"
    # So if input is a triplet, zero-indexed middle frame is 1
    # So if there's five frames as input, middle frame is 2
    middle_index = math.floor(num_frames_per_sample / 2)
    frame_nums = []

    for i in range(num_frames_per_sample):
        offset = abs(frame_delta * ((middle_index - i) ** 2))
        num_frame_num = -1
        if i < num_frames_per_sample / 2:
            num_frame_num = middle_frame - offset
            if clamp:
                num_frame_num = max(min_frame, num_frame_num)
        elif i == num_frames_per_sample / 2:
            num_frame_num = middle_frame
        else:
            num_frame_num = middle_frame + offset
            if clamp:
                num_frame_num = min(max_frames - 1, middle_frame + offset)

        frame_nums.append(num_frame_num)

    return frame_nums


def augment_char_crop(
    char_crop,
    horizontal_flip=0.5,
    hard_mode=0.1,
    downscale=0.2,
    resize=0.2,
    output_size=128,
    course_dropout=0.1,
    channel_dropout=0.0,
    pixel_dropout=0.1,
    gauss_noise=0.5,
):
    """
    Given a tight RGBA crop of a character, augment the image.

    Takes in a numpy data and returns numpy data.

    Crops that I want
    - horizontal flip
    - Brightness shift
    - color shift
    -
    """
    if output_size:
        char_crop = imutils.resize(char_crop, width=output_size)
        char_crop = np.array(
            ImageOps.pad(
                Image.fromarray(char_crop),
                (output_size, output_size),
                color=(0, 0, 0, 0),
            )
        )

    r, g, b = cv2.split(char_crop)
    rgb = cv2.merge((r, g, b))

    # These augmentations only work on 3 channels
    augmentations = [
        A.HorizontalFlip(p=horizontal_flip),
        # Make it brighter in order to capture the damage/iframe effect.
        A.RandomBrightnessContrast(p=0.3, brightness_limit=[-0.2, 0.4]),
        # A.Blur(blur_limit=[2, 5], p=0.2),
        # A.ChannelShuffle(p=1.0),
        A.Blur(blur_limit=[2, 3], p=0.05),
        A.HueSaturationValue(
            hue_shift_limit=[-256, 256],
            sat_shift_limit=[-67, 67],
            val_shift_limit=[-5, 5],
            p=1.0,
        ),
        A.GaussNoise(
            p=gauss_noise,
            var_limit=200,
            per_channel=True,
            mean=0.0,
        ),
    ]

    augmentations.append(
        A.PixelDropout(
            p=pixel_dropout,
            dropout_prob=random.uniform(0.0, 0.3),
            per_channel=False,
            drop_value=(0, 0, 0),
            mask_drop_value=0,
        )
    )
    augmentations.append(
        A.CoarseDropout(
            p=course_dropout,
            max_holes=int(random.uniform(1, 8)),
            max_height=min(4, int(char_crop.shape[0] / 8)),
            max_width=min(4, int(char_crop.shape[1] / 8)),
            min_holes=1,
            min_height=min(4, int(char_crop.shape[0] / 8)),
            min_width=min(4, int(char_crop.shape[1] / 8)),
            fill_value=(0, 0, 0),
            mask_fill_value=0,
        ),
    )
    augmentations.append(
        A.ChannelDropout(
            p=channel_dropout,
            channel_drop_range=(1, 2),
            fill_value=0,
        ),
    )

    augmentations.append(A.Downscale(p=downscale, scale_min=0.7, scale_max=0.9))

    if resize and output_size:
        # max_size = int(output_size * 1.2)
        augmentations.append(
            A.RandomSizedCrop(
                p=resize,
                min_max_height=(int(output_size * 0.7), output_size - 2),
                # min_max_height=(int(output_size * 0.7), int(output_size * 1.2) - 2),
                height=output_size,
                width=output_size,
                w2h_ratio=1.0,
                interpolation=0,
            )
        )

    transform = A.Compose(augmentations)
    transformed = transform(image=rgb)
    image_t = transformed["image"]

    if output_size:
        image_t = imutils.resize(image_t, width=output_size)

    return image_t


def augment_synth_char_crop(
    char_crop,
    horizontal_flip=0.5,
    hard_mode=0.1,
    downscale=0.2,
    resize=0.2,
    output_size=128,
):
    """
    Given a tight RGBA crop of a character, augment the image.

    Takes in a numpy data and returns numpy data.

    Crops that I want
    - horizontal flip
    - Brightness shift
    - color shift
    -
    """
    if output_size:
        char_crop = imutils.resize(char_crop, width=output_size)
        char_crop = np.array(
            ImageOps.pad(
                Image.fromarray(char_crop),
                (output_size, output_size),
                color=(0, 0, 0, 0),
            )
        )

    if resize and output_size:
        # A percent of the time, scale the character down to simulate the crop being too large.
        if random.random() > 0.6:
            new_scale = int(output_size * random.uniform(0.75, 1.0))
            char_crop = imutils.resize(char_crop, width=new_scale)
            char_crop = np.array(
                ImageOps.expand(
                    Image.fromarray(char_crop),
                    border=output_size - new_scale,
                    fill=(0, 0, 0, 0),
                )
            )

    r, g, b, a = cv2.split(char_crop)
    rgb = cv2.merge((r, g, b))

    # These augmentations only work on 3 channels
    augmentations = [
        A.HorizontalFlip(p=horizontal_flip),
        # Make it brighter in order to capture the damage/iframe effect.
        A.RandomBrightnessContrast(p=0.3, brightness_limit=[-0.2, 0.6]),
        # A.Blur(blur_limit=[2, 5], p=0.2),
        # A.ChannelShuffle(p=1.0),
        A.Blur(blur_limit=[2, 3], p=0.05),
        A.HueSaturationValue(
            hue_shift_limit=[-256, 256],
            sat_shift_limit=[-67, 67],
            val_shift_limit=[-10, 10],
            p=1.0,
        ),
        A.GaussNoise(
            p=0.2,
            var_limit=(427.63, 500.0),
            per_channel=True,
            mean=0.0,
        ),
    ]

    augmentations.append(
        A.PixelDropout(
            p=hard_mode,
            dropout_prob=0.1,
            per_channel=False,
            drop_value=(0, 0, 0),
            mask_drop_value=0,
        )
    )
    augmentations.append(
        A.CoarseDropout(
            p=hard_mode,
            max_holes=2,
            max_height=min(96, int(char_crop.shape[0] / 3)),
            max_width=min(96, int(char_crop.shape[1] / 3)),
            min_holes=1,
            min_height=min(96, int(char_crop.shape[0] / 3)),
            min_width=min(96, int(char_crop.shape[1] / 3)),
            fill_value=(0, 0, 0),
            mask_fill_value=0,
        ),
    )
    augmentations.append(
        A.ChannelDropout(
            p=hard_mode,
            channel_drop_range=(1, 2),
            fill_value=0,
        ),
    )

    augmentations.append(A.Downscale(p=downscale, scale_min=0.7, scale_max=0.9))

    if resize and output_size:
        # max_size = int(output_size * 1.2)
        augmentations.append(
            A.RandomSizedCrop(
                p=resize,
                min_max_height=(int(output_size * 0.3), output_size - 2),
                # min_max_height=(int(output_size * 0.7), int(output_size * 1.2) - 2),
                height=output_size,
                width=output_size,
                w2h_ratio=1.0,
                interpolation=0,
            )
        )

    transform = A.Compose(augmentations)
    transformed = transform(image=rgb, mask=a)
    image_t = transformed["image"]
    alpha_t = transformed["mask"]
    rgba = cv2.cvtColor(image_t, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = alpha_t

    if output_size:
        rgba = imutils.resize(rgba, width=output_size)

    return rgba


def get_anim_triplet_with_label(char_anim_dict, frame_delta, stage_paths, img_dimension):
    return None
    # char = random.choice(CHAR_LIST)
    # char_label = CHAR_LIST.index(char)

    # action = random.choice(list(char_anim_dict[char].keys()))
    # num_frames = len(char_anim_dict[char][action])
    # frame_num = random.randint(0, num_frames - 1)
    # frame_num_pre = max(0, frame_num - frame_delta)
    # frame_num_post = min(num_frames - 1, frame_num + frame_delta)
    # # TODO: allow the pre/post frames to be from different animations

    # stage = Image.open(random.choice(stage_paths))
    # stage_cropped = random_crop_pil_image(stage, img_dimension)

    # # TODO: add more synthetic alterations to the image.
    # frame_num = random.randint(0, num_frames - 1)
    # frame_num_pre = max(0, frame_num - frame_delta)
    # frame_num_post = min(num_frames - 1, frame_num + frame_delta)


def get_stage_paths():
    """
    Returns array of absolute paths to stages images.
    They seem to all be 1280 x 720
    """
    return glob(os.path.join(ULT_STAGES_DIR, "**/*.jpg"))


def get_character_animations_dict():
    """
    {
        "char_1": []
        "char_2": []
    }

    Animation frames are not sorted.
    """
    character_animations = {}
    for fighter in os.listdir(ULT_DATASET_CLEAN_CHAR_DIR):
        fighter_dir = os.path.join(ULT_DATASET_CLEAN_CHAR_DIR, fighter)
        if not os.path.isdir(fighter_dir):
            continue

        character_animations[fighter] = glob(os.path.join(fighter_dir, "*/*.png"))
    return character_animations


def get_character_actions_animations_dict():
    """
    {
        "char": {
            "animation_type": {
                "raw_anim": {
                    # (90,-90)
                    "cam_direction": [
                        # These frames will be sorted.
                        /abs/path/to/frame_0,
                        /abs/path/to/frame_1,
                        ...
                    ]
                }
            }
        }
    }
    """
    character_animations = {}
    for fighter in os.listdir(ULT_DATASET_CLEAN_CHAR_DIR):
        fighter_dir = os.path.join(ULT_DATASET_CLEAN_CHAR_DIR, fighter)
        if not os.path.isdir(fighter_dir):
            continue

        if fighter not in character_animations:
            character_animations[fighter] = {}

        for move in os.listdir(fighter_dir):
            move_dir = os.path.join(fighter_dir, move)
            if not os.path.isdir(move_dir):
                continue

            if move not in character_animations[fighter]:
                character_animations[fighter][move] = {}

            animation_files = glob(os.path.join(move_dir, "*.png"))

            for animation_file in animation_files:
                file_name = Path(animation_file).stem
                # Usually the pattern is '{char}_{body_type}_{anim_name}_frame_{cam}_{frame_num}.png
                # But the {anim_name} can be odd like:
                # 'byleth_c00_j02win1+us_en_frame_-90_63'

                # (
                #     _,
                #     body_type,
                #     anim_name,
                #     frame,
                #     cam,
                #     frame_num,
                # )
                anim_attributes = file_name.split("_")
                body_type = anim_attributes[1]
                cam = anim_attributes[-2]
                anim_name = "_".join(anim_attributes[2:-2])

                if body_type not in character_animations[fighter][move]:
                    character_animations[fighter][move][body_type] = {}
                if anim_name not in character_animations[fighter][move][body_type]:
                    character_animations[fighter][move][body_type][anim_name] = {}
                if cam not in character_animations[fighter][move][body_type][anim_name]:
                    character_animations[fighter][move][body_type][anim_name][cam] = []

                character_animations[fighter][move][body_type][anim_name][cam].append(
                    animation_file
                )

            # Now we need to sort each of the animation_files.  Sorting by string won't work due to
            # bad original naming.
            for body_type in character_animations[fighter][move]:
                for anim_name in character_animations[fighter][move][body_type]:
                    for cam in character_animations[fighter][move][body_type][anim_name]:
                        character_animations[fighter][move][body_type][anim_name][cam] = sorted(
                            character_animations[fighter][move][body_type][anim_name][cam],
                            key=lambda x: int(Path(x).stem.split("_")[-1]),
                        )

    return character_animations
