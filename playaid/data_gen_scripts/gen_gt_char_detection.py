"""
Converts ground truth data into yolov5 character detection.
"""
import cv2
import os
from tqdm import tqdm
from playaid.constants import (
    CHAR_LIST,
    GROUND_TRUTH_CHAR_DETECTION_DIR,
    GROUND_TRUTH_TRAIN,
    GROUND_TRUTH_VAL,
    GROUND_TRUTH_TEST,
    GROUND_TRUTH_DIR,
)

from playaid.timeline import (
    load_ground_truth_pairings_from_file,
    load_ground_truth_from_path,
    update_fighters_from_timeline,
)


def write_yolo_output(output_path, yolo_data):
    """
    Writes bbox data to output_path
    @param output_path: .txt path
    @param yolo_data: array of yolo class_id, bbox data.  [(class_id, bbox_data), etc]
    """
    with open(output_path, "w") as f:
        for (class_id, bbox_yolo) in yolo_data:
            # a normalized square crop won't have matching normalized width/height since the
            # overall image has a different width/height.
            f.write(f"{class_id} {bbox_yolo[0]} {bbox_yolo[1]} {bbox_yolo[2]} {bbox_yolo[3]}\n")


def generate_data(
    pairings_file, sub_dir_name, interval=1, offset=0, max_frames=None, overwrite=False
):
    """ """
    sub_dir = os.path.join(GROUND_TRUTH_CHAR_DETECTION_DIR, sub_dir_name)
    images_dir = os.path.join(sub_dir, "images")
    labels_dir = os.path.join(sub_dir, "labels")

    if not os.path.exists(sub_dir):
        os.makedirs(images_dir)
        os.makedirs(labels_dir)

    pairings = load_ground_truth_pairings_from_file(pairings_file)
    for pairing in pairings:
        dir_name, video_name, log_name, log_offset = pairing
        video_path = os.path.join(GROUND_TRUTH_DIR, dir_name, video_name)
        label_path = os.path.join(GROUND_TRUTH_DIR, dir_name, log_name)
        video = cv2.VideoCapture(video_path)
        max_frames = max_frames if max_frames else int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        timeline = load_ground_truth_from_path(label_path, log_offset=log_offset)

        fighters = []

        print(f"{sub_dir.capitalize()} - {video_path}")
        for i in tqdm(range(offset, max_frames)):
            if i >= len(timeline):
                break

            if max_frames and i >= max_frames:
                break

            if (i + offset) % interval != 0:
                continue

            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            res, input_frame = video.read()
            if not res:
                break

            fighters = update_fighters_from_timeline(i, timeline[i], fighters)

            yolo_data = [
                (CHAR_LIST.index(f.fighter_name), (f.crop.square_yolo_crop(input_frame)))
                for f in fighters
            ]

            output_img_path = os.path.join(images_dir, f"{dir_name}-{i}.jpg")

            if not overwrite and os.path.exists(output_img_path):
                print(f"Already created the images for {dir_name}")
                break

            cv2.imwrite(output_img_path, input_frame)

            output_label_path = os.path.join(labels_dir, f"{dir_name}-{i}.txt")
            write_yolo_output(output_label_path, yolo_data)


if __name__ == "__main__":
    generate_data(GROUND_TRUTH_TRAIN, "train", interval=5)
    generate_data(GROUND_TRUTH_VAL, "validation", interval=60 * 10, offset=3)
    generate_data(GROUND_TRUTH_TEST, "test", interval=60 * 15, offset=6)

    print("🎉 COMPLETED 🎉")
