import os
import click
from datetime import datetime
from pathlib import Path
import cv2

# from action_detector import RNNActionDetector
import shutil
from tqdm import tqdm
from moviepy.editor import VideoFileClip
import subprocess

from playaid.annotator import Annotator
from playaid.stats import Stats, get_stats_at_frame
from playaid.timeline import (
    load_ground_truth_from_path,
    load_ground_truth_pairings_from_file,
    update_fighters_from_timeline,
    load_timeline_from_ai_output,
)
from playaid import constants


class Manuscript:
    """
    Runs e2e tracking and action recognition.
    """

    def __init__(
        self,
        input_video_path: str = os.path.join(
            constants.ULT_DATASET_DIR, "ult_videos/tweek-mkleo-clip.mp4"
        ),
        output_video_path: str = os.path.join(
            constants.EXPERIMENT_OUTPUT, "tweek-mkleo-clip-manuscript.mp4"
        ),
        start_frame: int = 0,
        max_frames: int = -1,
        image_debug=False,
        action_detection_output=None,
        ground_truth_path=None,
        ai_output_path=None,
        # This will change, this was what I trained the above checkpoints on
        actions=["ForwardSmash", "NeutralAir", "BackAir", "Dash", "Unknown"],
        skip_graphs: bool = False,
        log_offset: int = 0,
        include_audio: bool = True,
        skip_summaries: bool = False,
        run_ai: bool = False,
        show_timer: bool = False,
    ):
        """
        @param yolo_output_dir
        @param action_detection_output: path to the output directory of the action recognition model
        so that it doesn't need to rerun each time.
        """
        self.stats = Stats(input_video_path)
        self.output_video_path = output_video_path
        output_video_path = Path(output_video_path)
        self.debug_output_dir = os.path.join(
            os.path.dirname(output_video_path.absolute()), output_video_path.stem
        )

        self.ai_output_path = ai_output_path
        self.image_debug = image_debug
        if os.path.exists(self.debug_output_dir):
            shutil.rmtree(self.debug_output_dir)

        self.input_video_path = input_video_path
        self.input_video = cv2.VideoCapture(input_video_path)
        self.fps = self.input_video.get(cv2.CAP_PROP_FPS)
        # self.fps = 60.0
        self.w = int(self.input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fighters = []
        self.log_offset = log_offset

        # Load the models from the checkpoints
        self.action_detect_models = {}

        self.start_frame = start_frame
        if max_frames < 0:
            max_frames = int(self.input_video.get(cv2.CAP_PROP_FRAME_COUNT))

        self.max_frames = max_frames
        print(f"max frames: {int(self.input_video.get(cv2.CAP_PROP_FRAME_COUNT))}, fps: {self.fps}")

        self.action_detection_output = action_detection_output
        self.skip_graphs = skip_graphs
        self.include_audio = include_audio
        self.skip_summaries = skip_summaries
        self.ground_truth_path = ground_truth_path
        self.show_timer = show_timer

        self.subtitle = "Placeholder"
        # Collects hashes that it hasn't seen to be printed out later
        self.unknown_hashes = set()

        if ground_truth_path:
            self.timeline = load_ground_truth_from_path(ground_truth_path, log_offset=log_offset)
        if self.ai_output_path:
            self.timeline = load_timeline_from_ai_output(ai_output_path)

    def update_fighters_from_gt(self, frame_number: int):
        if frame_number >= len(self.timeline):
            return False

        update_fighters_from_timeline(frame_number, self.timeline[frame_number], self.fighters)
        return True

    def render(self):
        """
        Renders the visualization.
        """
        colors = {
            0: (25, 58, 115),
            1: (201, 99, 48),
            2: (201, 99, 48),
            3: (201, 99, 48),
            4: (201, 99, 48),
            5: (201, 99, 48),
            6: (201, 99, 48),
            7: (201, 99, 48),
        }
        # # Create per image based output as well for debugging purposes.
        # if not os.path.exists(self.debug_output_dir):
        #     os.makedirs(self.debug_output_dir)

        show_stats = not self.skip_graphs
        annotator = Annotator(
            self.output_video_path,
            int(self.fps),
            self.w,
            self.h,
            show_stats=show_stats,
        )

        # Even if we aren't rendering, run the common operations the frame's leading up to the
        # rendered frames.
        if self.start_frame:
            print(f"Running frames 0 to {self.start_frame}")
            for i in tqdm(range(self.start_frame)):
                if not self.update_fighters_from_gt(i):
                    break

                self.stats.record_frame(self.fighters)

        # I have no clue why, but if I want to start a video not at 0 I have to do this or the
        # video will be at a completely wrong frame.
        self.input_video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        print(f"Rendering frames {self.start_frame} to {self.max_frames}")
        for i in tqdm(range(self.start_frame, self.max_frames)):
            self.input_video.set(cv2.CAP_PROP_POS_FRAMES, i)
            res, input_frame = self.input_video.read()
            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGBA)
            assert res, f"Failed to read frame {i} during Manuscript render"
            annotator.set_frame(input_frame, line_width=4, font_size=0.2, pil=False)

            if not self.update_fighters_from_gt(i):
                break

            self.stats.record_frame(self.fighters)

            for j, fighter in enumerate(self.fighters):
                if self.log_offset < 0 and i < abs(self.log_offset):
                    break

                label = (
                    f"{fighter.action}"
                    if fighter.action != "Undefined" and fighter.action != ""
                    # else f"{fighter.action_string} | {fighter.motion_hex}"
                    else ""
                )

                if True:
                    label += f" | #{fighter.animation_frame_num}"
                if True and fighter.anim_state:
                    label += f" | {fighter.anim_state}"
                # if True:
                #     label += f" | {fighter.status}"

                if fighter.action == "Undefined" or not fighter.action:
                    if fighter.motion_hex not in self.unknown_hashes:
                        print(
                            f"Unknown hex for {fighter.fighter_name} at {i} - {fighter.motion_hex}"
                        )
                        self.unknown_hashes.add(fighter.motion_hex)

                other_fighter = self.fighters[(j + 1) % len(self.fighters)]

                if (
                    False
                    and fighter.frames_since_damaged < 10
                    and other_fighter.previous_non_damaged_action
                ):
                    label += f" <-- {other_fighter.previous_non_damaged_action}"

                color = colors[fighter.fighter_id]
                if fighter.hitstun_left:
                    # Make the background gray if a character is in hitstun.
                    color = (55, 55, 55)

                annotator.box_label(
                    fighter.crop.xyxy_pixels(input_frame.shape[1], input_frame.shape[0]),
                    label=label,
                    color=color,
                    draw_box=False,
                )

                # Show time_remaining in the top right.
                if False:
                    annotator.box_label(
                        (980, 30 + (j * 50), 1200, 60),
                        label=f"#{i} - {fighter.time_remaining}",
                        color=colors[fighter.fighter_id],
                        draw_box=False,
                    )

            # Show time_remaining in the top right.
            if self.show_timer:
                annotator.box_label(
                    (980, 80, 1200, 60),
                    label=f"Frame #{max(i + self.log_offset, 0)}",
                    color=colors[fighter.fighter_id],
                    draw_box=False,
                )

            annotator.update_onscreen_charts(self.fighters, self.stats)

            if show_stats:
                annotator.update_offscreen_charts(self.fighters, self.stats)

            rendered_result = annotator.result()
            annotator.write()

            if self.image_debug:
                debug_image_path = os.path.join(self.debug_output_dir, f"{i}.png")
                cv2.imwrite(debug_image_path, rendered_result)

        if not self.skip_summaries:
            annotator.post_game_summaries(self.fighters, self.stats)

        annotator.video_writer.release()

        if self.include_audio and self.start_frame == 0:
            self.add_audio()

    def add_audio(self):
        """
        Adds audio from the original video the output video.
        """
        # Add audio using moviepy
        print("Adding audio")
        vid_with_audio_tmp_path = os.path.join("/tmp", Path(self.output_video_path).name)
        command = [
            "ffmpeg",
            "-i",
            self.output_video_path,
            "-i",
            self.input_video_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-strict",
            "experimental",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-shortest",
            vid_with_audio_tmp_path,
        ]

        subprocess.run(command, check=True)

        # Ovewrite the video that doesn't have audio with the one that does.
        shutil.move(vid_with_audio_tmp_path, self.output_video_path)

    def __str__(self):
        """ """
        repr = []
        for i in range(len(self.timeline)):
            chars = self.timeline[i]
            # self.raw_yolo_labels, key=lambda x: int(Path(x).stem.split("_")[-1])
            chars = sorted(chars, key=lambda c: c.fighter_name)
            repr.append(f"{i} - {[str(c) for c in chars]}")

        return "\n".join(repr)


@click.command()
@click.option(
    "--frames",
    "-f",
    default=None,
    help="Frames in the format start,end. If empty, will use entire video.",
)
@click.option(
    "--skip-graphs",
    "-s",
    is_flag=True,
    help="Whether to skip the graphs on the sides of the video (faster)",
)
@click.option(
    "--video-index", "-v", default=None, help="Index of the video you want to play from train.csv"
)
@click.option("--skip-summaries", "-c", is_flag=True, help="If true, skip post-game summary")
@click.option("--show-timer", "-t", is_flag=True, help="Show timer in top right for debugging")
@click.option("--video-path", "-p", default=None, help="Path to input video")
@click.option("--log-path", "-p", default=None, help="Path to the input log")
@click.option("--ai-output-path", "-ai", default=None, help="Path to cached ai output")
def run_manuscript(
    frames,
    skip_graphs,
    video_index,
    skip_summaries,
    show_timer,
    video_path,
    log_path,
    ai_output_path,
):
    """Entrypoint to Manuscript"""
    if not video_index and not video_path:
        print("Must specify either --video-index or --video-path")
        return

    # Get the current date and time and convert to a string
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d-%H-%M-%S")

    start_frame, end_frame = 0, -1
    if frames:
        start_frame, end_frame = map(int, frames[1:].split(","))

    manuscript_args = {
        "start_frame": start_frame,
        "max_frames": end_frame,
        "skip_graphs": skip_graphs,
        "include_audio": True,
        "skip_summaries": skip_summaries,
        "show_timer": show_timer,
    }

    if video_index:
        pairings = load_ground_truth_pairings_from_file(constants.GROUND_TRUTH_TRAIN)
        dir_name, video_name, log_name, log_offset = pairings[int(video_index)]
        video_path = os.path.join(constants.GROUND_TRUTH_DIR, dir_name, video_name)
        label_path = os.path.join(constants.GROUND_TRUTH_DIR, dir_name, log_name)
        manuscript_args.update(
            {
                "input_video_path": video_path,
                "output_video_path": os.path.join(
                    constants.EXPERIMENT_OUTPUT,
                    f"{dir_name}-{start_frame}-{end_frame}_{date_time_str}.mp4",
                ),
                "ground_truth_path": label_path,
                "log_offset": log_offset,
            }
        )

    else:
        _, file_name = os.path.split(video_path)
        video_name, _ = os.path.splitext(file_name)
        manuscript_args.update(
            {
                "input_video_path": video_path,
                "ai_output_path": ai_output_path,
                # only run the AI if we don't already have the output cached.
                "run_ai": not bool(ai_output_path),
                "ground_truth_path": log_path,
                "output_video_path": os.path.join(
                    constants.EXPERIMENT_OUTPUT,
                    f"{video_name}-{start_frame}-{end_frame}_{date_time_str}.mp4",
                ),
                "log_offset": 5,
            }
        )

    manuscript = Manuscript(**manuscript_args)

    manuscript.render()

    print("🎉 COMPLETED 🎉")


if __name__ == "__main__":
    run_manuscript()
