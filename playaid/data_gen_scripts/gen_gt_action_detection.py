"""
Converts ground truth data into yolov5 character detection.
"""
import cv2
import os
from playaid.constants import (
    ACTION_GROUND_TRUTH_DIR,
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

from multiprocessing import Pool, cpu_count


OVERWRITE = False


def process_pairing(args):
    print("args: ", args)
    sub_dir, pairing = args
    dir_name, video_name, log_name, log_offset = pairing
    video_path = os.path.join(GROUND_TRUTH_DIR, dir_name, video_name)
    label_path = os.path.join(GROUND_TRUTH_DIR, dir_name, log_name)
    video = cv2.VideoCapture(video_path)
    max_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    timeline = load_ground_truth_from_path(label_path, log_offset=log_offset)

    fighters = []

    for i in range(max_frames):
        if i >= len(timeline):
            break

        if max_frames and i >= max_frames:
            break
 
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        res, input_frame = video.read()
        fighters = update_fighters_from_timeline(i, timeline[i], fighters)

        should_break = False

        for j, fighter in enumerate(fighters):
            output_dimension = 128
            res, crop = fighter.crop.square_crop(input_frame, output_dimension, padding=30)
            if not res:
                # Character must be offscreen.
                continue

            anim_dir = os.path.join(
                sub_dir,
                dir_name,
                f"{fighter.fighter_id}_{fighter.fighter_name.lower().replace(' ', '_')}",
            )

            if i == 0 and j == 0 and os.path.exists(anim_dir):
                print(f"Already created the data for {dir_name}")
                should_break = True
                break

            labels_dir = os.path.join(anim_dir, "labels")
            images_dir = os.path.join(anim_dir, "images")

            if not os.path.exists(anim_dir):
                os.makedirs(images_dir)
                os.makedirs(labels_dir)

            output_img_path = os.path.join(
                images_dir,
                f"{str(i).zfill(6)}.jpg",
            )
            cv2.imwrite(output_img_path, crop)

            output_label_path = os.path.join(
                labels_dir,
                f"{str(i).zfill(6)}.txt",
            )

            with open(output_label_path, "w") as f:
                f.write(fighter.action or "Undefined")

        if should_break:
            break


def generate_data(pairings_file, sub_dir_name):
    """
    Output directory will be
    """
    sub_dir = os.path.join(ACTION_GROUND_TRUTH_DIR, sub_dir_name)

    pairings = load_ground_truth_pairings_from_file(pairings_file)
    num_processes = cpu_count() - 4
    print(f"num_processes: {num_processes}")

    with Pool(num_processes) as p:
        p.map(process_pairing, [(sub_dir, pairing) for pairing in pairings])

    # for pairing in pairings:
    #     process_pairing((sub_dir, pairing))


if __name__ == "__main__":
    generate_data(GROUND_TRUTH_TRAIN, "train")
    generate_data(GROUND_TRUTH_VAL, "validation")
    generate_data(GROUND_TRUTH_TEST, "test")

    print("🎉 COMPLETED 🎉")
