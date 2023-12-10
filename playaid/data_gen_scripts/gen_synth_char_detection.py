"""
Generates composites of characters on the stages for YoloV5 character detection.
"""
import cv2 as cv
from glob import glob
import numpy as np
import os
from PIL import Image
import subprocess
import random

from playaid.constants import (
    CHAR_LIST,
    ULT_DATASET_CLEAN_CHAR_DIR,
    COMPOSITES_DIR,
)
from playaid.dataset_utils import (
    get_stage_paths,
    get_character_actions_animations_dict,
    augment_synth_char_crop,
)
from playaid.data_gen_scripts.raw_anim_data_cleaner import get_bounding_box
from playaid.anim_ontology import MOVE_TO_CLASS_ID

MAX_NUM_CHAR = 4

# 'CHAR' means the label will just be the character.  Used for Yolo character detection.
# 'CHAR+ACTION' means the label will just be specific for each character x action combination.  Used
# for Yolo-based action detection which likely will not work well.
CLASS_TYPE = "CHAR+ACTION"


def bbox_yolo_to_corners(bbox):
    """
    Converts bounding box representation from yolo to corners
    """
    x_center, y_center, width, height = bbox
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


def bbox_corners_to_yolo(bbox):
    """
    Converts bounding box representation from corners to yolo

    Yolo bbox style is x_center y_center width height.
    """
    ul, _, _, br = bbox
    left, top = ul
    right, bottom = br
    x_center = left + ((right - left) / 2)
    y_center = top + ((bottom - top) / 2)
    width = right - left
    height = bottom - top
    return (int(x_center), int(y_center), int(width), int(height))


def write_yolo_output(output_path, yolo_data):
    """
    Writes bbox data to output_path
    @param output_path: .txt path
    @param yolo_data: array of yolo class_id, bbox data.  [(class_id, bbox_data), etc]
    """
    with open(output_path, "w") as f:
        for (class_id, bbox_yolo) in yolo_data:
            f.write(f"{class_id} {bbox_yolo[0]} {bbox_yolo[1]} {bbox_yolo[2]} {bbox_yolo[3]}\n")


def read_yolo_ouput(output_path):
    yolo_output = []
    with open(output_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            bbox = [int(x) for x in line.split(" ")[1:]]
            class_id = int(line.split(" ")[0])
            yolo_output.append((class_id, bbox))

    return yolo_output


def create_bounding_box_file(img, path, overwrite=False):
    """
    Creates a .txt file for the YOLO spec.
    https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#12-create-labels-1
    Each row is:
     - class x_center y_center width height

    Returns data in yolo representation
    """
    if not overwrite and os.path.exists(path):
        read_yolo_ouput(path)[0]

    bbox = get_bounding_box(img)
    bbox_yolo = bbox_corners_to_yolo(bbox)
    character = path.split("/")[-3]
    class_id = CHAR_LIST.index(character)
    write_yolo_output(path, [(class_id, bbox_yolo)])

    return class_id, bbox_yolo


def clean_ultimate_frame_data_image(img):
    """
    Removes gray background color from image.  Input is RGB, output is RGBA.
    """
    cv.cvtColor(img, cv.COLOR_RGB2HSV)

    mask = 255 - cv.inRange(img, np.array([62, 62, 62]), np.array([65, 65, 65]))
    r, g, b = cv.split(img)
    cleaned = cv.merge([r, g, b, mask], 4)

    return cleaned


def write_cleaned_ultime_frame_data_gif(gif_path, overwrite=False):
    """
    Converts a gif into a series of cleaned, cropped images
    DEPRECATED
    """
    # gif = imageio.mimread(gif_path, memtest="1000MB")
    gif = Image.open(gif_path)
    nums = gif.n_frames
    print(f"Total {nums} frames in the ${gif_path}!")

    character, animation = gif_path.split("/")[-2:]
    animation_name = animation.split(".")[0]
    char_dir = os.path.join(ULT_DATASET_CLEAN_CHAR_DIR, character, animation_name)
    if not os.path.exists(char_dir):
        os.makedirs(char_dir)

    raw_output = os.path.join(char_dir, f"{animation_name}-%d.png")
    # ImageMagick convert is the only thing I've used that worked across PC and Mac
    # in terms of properly converting a gif into a series of images.
    if not os.path.exists(os.path.join(char_dir, f"{animation_name}-0.png")):
        command = ["convert", gif_path, "-coalesce", raw_output]
        subprocess.run(command)

    for i in range(nums):
        # Read in RGB with grayscale background that was created by imagemagick
        input_path = os.path.join(char_dir, f"{animation_name}-{i}.png")

        # this is the uncropped RGBA.
        output_path = os.path.join(char_dir, f"{animation_name}-{i}-clean.png")
        if not os.path.exists(input_path):
            print(f"Missing input path {input_path}")
            continue

        img = cv.imread(input_path)
        if not os.path.exists(output_path) or overwrite:
            cleaned_image = clean_ultimate_frame_data_image(img)
            cv.imwrite(output_path, cleaned_image)
        else:
            # Needs IMREAD_UNCHANGED to read as rgba
            cleaned_image = cv.imread(output_path, cv.IMREAD_UNCHANGED)

        output_path = os.path.join(char_dir, f"{animation_name}-{i}.txt")
        class_id, bbox = create_bounding_box_file(cleaned_image, output_path, overwrite=overwrite)
        bbox = bbox_yolo_to_corners(bbox)
        top_left, _, _, bottom_right = bbox
        top, left = top_left
        bottom, right = bottom_right

        output_path = os.path.join(char_dir, f"{animation_name}-{i}-bbox.png")
        if not os.path.exists(output_path) or overwrite:
            bbox_image = cv.rectangle(
                # I had to copy the image otherwise the cropped output was getting
                # the bounding box on PC.
                cleaned_image.copy(),
                top_left,
                bottom_right,
                (255, 0, 0, 255),
                thickness=10,
            )
            cv.imwrite(output_path, bbox_image)

        output_path = os.path.join(char_dir, f"{animation_name}-{i}-crop.png")
        if not os.path.exists(output_path) or True:
            cropped_image = cleaned_image[left:right, top:bottom]
            cv.imwrite(output_path, cropped_image)


def composite_chars_onto_stage(i, stage_path, char_paths, output_path, bbox_overlay=False):
    """
    @param i: debugging index
    @param stage_path: string path
    @param char_paths: array of string paths to images of a character.
    @param bbox_overlay: bool to add bounding box onto the image
    """
    stage = Image.open(stage_path)

    # [(class_id, normalized_bbox), ...]
    yolo_output = []
    pixel_bbox_data = []
    for char_path in char_paths:
        char = Image.open(char_path)
        # print(f"#{i} {char.width}w x {char.height}h | {char_path}")

        # TODO: clean this up on the frontend
        if char.width < 100 or char.height < 100:
            continue

        char_label = CHAR_LIST.index(os.path.dirname(char_path).split(os.sep)[-2])
        action_label = MOVE_TO_CLASS_ID[os.path.dirname(char_path).split(os.sep)[-1]]
        char_and_action_label = len(MOVE_TO_CLASS_ID.keys()) * char_label + action_label
        class_id = char_label if CLASS_TYPE == "CHAR" else char_and_action_label

        # Resize characters so that the max width is 100 pixels
        basewidth = random.randint(50, 150)
        wpercent = basewidth / float(char.size[0])
        hsize = int((float(char.size[1]) * float(wpercent)))
        char = char.resize((basewidth, hsize))

        char_np = augment_synth_char_crop(np.array(char))
        char = Image.fromarray(char_np)

        # 99.73% chance the sample will fall within desired range with normal distribution towards
        # center.
        center_x = int(random.gauss(stage.width / 2, stage.width / 6))
        center_y = int(random.gauss(stage.height / 2, stage.height / 6))
        # Just put it in the middle if the image falls outside the expected range.
        if center_x < 0 or center_x > stage.width:
            center_x = stage.width / 2
        if center_y < 0 or center_y > stage.height:
            center_y = stage.height / 2

        stage.paste(
            char,
            # paste just wants upper left coordinate.
            (int(center_x - (char.width / 2)), int(center_y - (char.height / 2))),
            # Char is also used as the transparency mask
            char,
        )

        # Store pixel bbox data for visualizations.
        pixel_bbox_data.append((center_x, center_y, char.width, char.height))

        # We need to convert from pixel to normalized [0, 1] range
        center_x = center_x / stage.width
        center_y = center_y / stage.height
        width = char.width / stage.width
        height = char.height / stage.height
        normalized_yolo_bbox = (center_x, center_y, width, height)
        yolo_output.append((class_id, normalized_yolo_bbox))

    stage = cv.cvtColor(np.array(stage), cv.COLOR_RGB2BGR)
    if bbox_overlay:
        for pixel_bbox in pixel_bbox_data:
            top_left, _, _, bottom_right = bbox_yolo_to_corners(pixel_bbox)
            stage = cv.rectangle(stage, top_left, bottom_right, (255, 0, 0, 255), thickness=4)

    cv.imwrite(output_path, stage)

    output_path = output_path.replace("images", "labels").replace(".jpg", ".txt")
    write_yolo_output(output_path, yolo_output)


def generate_stage_char_compositions(
    sub_dir_name, n_generations, log_interval=50, overwrite=False, bbox_overlay=False
):
    """
    Generates n_generations compositions of characters on a smash ultimate stage.
    """
    stages = get_stage_paths()
    char_animations = get_character_actions_animations_dict()

    sub_dir = os.path.join(COMPOSITES_DIR, sub_dir_name)
    images_dir = os.path.join(sub_dir, "images")
    labels_dir = os.path.join(sub_dir, "labels")
    if not os.path.exists(sub_dir):
        os.makedirs(images_dir)
        os.makedirs(labels_dir)

    # If overwrite == True, then we'll overwrite the existing files.
    num_existing_images = 0 if overwrite else len(glob(os.path.join(images_dir, "*.jpg")))

    for i in range(num_existing_images, num_existing_images + n_generations):
        if i and i % log_interval == 0:
            print(f"Generated {i} composites for {sub_dir_name}/")

        output_path = os.path.join(images_dir, f"comp-{i}.jpg")

        # Make sure the characters are evenly selected
        num_chars_in_composite = random.choice(range(1, MAX_NUM_CHAR + 1))
        selected_animations = []
        for i in range(num_chars_in_composite):
            character = random.choice(CHAR_LIST)
            action = random.choice(list(char_animations[character].keys()))
            selected_animations.append(random.choice(char_animations[character][action]))

        stage = random.choice(stages)

        composite_chars_onto_stage(i, stage, selected_animations, output_path, bbox_overlay)


if __name__ == "__main__":
    generate_stage_char_compositions(
        "train", 20000, log_interval=10, overwrite=False, bbox_overlay=False
    )
    generate_stage_char_compositions("validation", 256)
    generate_stage_char_compositions("test", 256)

    print("🎉 COMPLETED 🎉")
