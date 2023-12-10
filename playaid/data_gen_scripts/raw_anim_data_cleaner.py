"""
Cleans images (crops, removes background) in ULT_DATASET_RAW_CHAR_DIR and places them in
ULT_DATASET_CLEAN_CHAR_DIR.
It also renames them from the raw animation filename to the specified anim ontology.
"""

import cv2 as cv
import pathlib
import numpy as np
import os
from multiprocessing import Pool

from playaid.dataset_utils import get_animation_type_for_anim_file
from playaid.anim_ontology import ANIM_FILE_TO_ANIMATION
from playaid.constants import ULT_DATASET_CLEAN_CHAR_DIR, ULT_DATASET_RAW_CHAR_DIR

USE_MULTIPROCESSING = True


def get_bounding_box(img):
    """
    Very slow implementation.  Creates a bounding box for a single character in a cleaned image.
    The input image should be a RGBA numpy.

    Returns ul, ur, bl, br.
    """
    top = img.shape[1]
    bottom = 0
    left = img.shape[1]
    right = 0
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            pixel = img[x, y]

            # Checks alpha.
            if pixel[3] == 255:
                top = min(top, x)
                bottom = max(bottom, x)
                left = min(left, y)
                right = max(right, y)

    return ((left, top), (right, top), (left, bottom), (right, bottom))


def remove_black_background(img):
    """
    Removes black background color from image.  Input is RGB, output is RGBA.
    """
    cv.cvtColor(img, cv.COLOR_RGB2HSV)

    mask = 255 - cv.inRange(img, np.array([0, 0, 0]), np.array([1, 1, 1]))
    r, g, b = cv.split(img)
    cleaned = cv.merge([r, g, b, mask], 4)

    return cleaned


def clean_single_raw_fighter_anim_data(raw_image_path: str, fighter: str):
    """
    @param raw_image_path: absolute path to raw image.
    @param fighter: name of fighter.
    """
    img = cv.imread(raw_image_path)

    # Remove black background
    transparent_image = remove_black_background(img)

    # Create a tight crop of the animation.
    bbox = get_bounding_box(transparent_image)
    top_left, _, _, bottom_right = bbox
    top, left = top_left
    bottom, right = bottom_right
    cropped_image = transparent_image[left:right, top:bottom]
    return cropped_image


def clean_raw_fighter_anim_data(
    fighter: str, raw_animation_name: str, limit=None, overwrite=False
):
    fighter_dir = os.path.join(ULT_DATASET_RAW_CHAR_DIR, fighter)
    animation_type = get_animation_type_for_anim_file(raw_animation_name)
    if animation_type == "Undefined":
        print(
            f"Skipping {raw_animation_name} for {fighter} since animation_type == 'Undefined'"
        )
        return

    output_dir = os.path.join(ULT_DATASET_CLEAN_CHAR_DIR, fighter, animation_type)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    input_animation_dir = os.path.join(fighter_dir, raw_animation_name)
    print(f"Working on {fighter} {animation_type} from {raw_animation_name}")

    for file in pathlib.Path(input_animation_dir).iterdir():
        if ".png" not in file.name:
            continue

        output_file = os.path.join(output_dir, file.name)

        if os.path.exists(output_file) and not overwrite:
            # We can assume we've done the rest
            break

        cropped_image = clean_single_raw_fighter_anim_data(str(file), fighter)

        # If the character left the frame, don't write anything.
        if not cropped_image.shape[0] or not cropped_image.shape[1]:
            continue

        cv.imwrite(output_file, cropped_image)

    print(f"Finished {fighter}'s {animation_type} from {raw_animation_name}")


def clean_all_raw_fighter_anim_data(fighter: str, limit=None, overwrite=False):
    """
    Converts all of a single fighter's raw anim images
    """
    clean_dir = os.path.join(ULT_DATASET_CLEAN_CHAR_DIR, fighter)
    if not os.path.exists(clean_dir):
        os.makedirs(clean_dir)

    fighter_dir = os.path.join(ULT_DATASET_RAW_CHAR_DIR, fighter)

    raw_animations = [
        d
        for d in os.listdir(fighter_dir)
        if os.path.isdir(os.path.join(fighter_dir, d))
    ]

    if USE_MULTIPROCESSING:
        anim_args = [
            (fighter, raw_anim, limit, overwrite) for raw_anim in raw_animations
        ]
        with Pool(8) as pool:
            pool.starmap(clean_raw_fighter_anim_data, anim_args)
    else:
        for raw_anim in raw_animations:
            clean_raw_fighter_anim_data(fighter, raw_anim, limit, overwrite)


def clean_raw_anim_data(limit=None):
    """
    Cleans images (crops, removes background) in ULT_DATASET_RAW_CHAR_DIR and places them in
    ULT_DATASET_CLEAN_CHAR_DIR.
    It also renames them from the raw animation filename to the specified anim ontology.
    """
    # Each character has a list of animation data that is valid.
    for fighter in os.listdir(ULT_DATASET_RAW_CHAR_DIR):
        print(f"Working on fighter: {fighter}")
        fighter_dir = os.path.join(ULT_DATASET_RAW_CHAR_DIR, fighter)
        if not os.path.isdir(fighter_dir):
            continue

        clean_all_raw_fighter_anim_data(fighter, limit)


if __name__ == "__main__":
    # Stage 1
    clean_raw_anim_data()
