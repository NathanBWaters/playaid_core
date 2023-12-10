import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib

from playaid.graphs.utils import (
    is_ascii,
    paste_on_top,
    split_text_emoji,
)
from playaid.graphs.onscreen import move_pie_chart, move_pie_chart_history
from playaid.graphs.timeline import (
    disadvantage_tech_history,
    disadvantage_ledge_history,
)
from playaid.graphs.bar_charts import (
    disadvantage_ledge_option_chart,
    disadvantage_tech_option_chart,
    move_damage_graph,
    move_success_punished_missed_bar_graph,
    defensive_option_chart,
)
from constants import TEXT_FONT_PATH, EMOJI_FONT_PATH


def split_text(text, chunk_size=90):
    words = text.split()
    chunks = []
    chunk = ""
    for word in words:
        if len(chunk) + len(word) <= chunk_size:
            chunk += " " + word if chunk else word
        else:
            chunks.append(chunk)
            chunk = word
    if chunk:
        chunks.append(chunk)
    return chunks


class Annotator:
    # Class called by Manuscript.py to created annotated video outputs.
    def __init__(
        self,
        output_video_path: str,
        fps: int,
        input_width: int,
        input_height: int,
        show_stats=True,
    ):
        """ """
        self.output_video_path = output_video_path
        self.fps = fps
        self.input_width = input_width
        self.input_height = input_height
        # Right padding adds a black portion on the right side.
        self.show_stats = show_stats

        self.right_padding = 0
        self.left_padding = 0
        self.bottom_padding = 0
        if self.show_stats:
            self.right_padding = 400
            self.left_padding = 400
            self.bottom_padding = 400

        # The output will be bigger than the input because it has the black padding.
        self.output_width = self.input_width + self.left_padding + self.right_padding
        self.output_height = self.input_height + self.bottom_padding
        # The class that handles writing frames to the output video.
        self.video_writer = cv2.VideoWriter(
            self.output_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.output_width, self.output_height),
        )

    def set_frame(
        self,
        im,
        line_width=None,
        font_size=None,
        font=TEXT_FONT_PATH,
        pil=False,
        example="abc✅",
    ):
        assert (
            im.data.contiguous
        ), "Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images."
        non_ascii = not is_ascii(example)  # non-latin labels, i.e. asian, arabic, cyrillic
        self.pil = pil or non_ascii
        if self.pil:  # use PIL
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            # self.text_font = ImageFont.truetype(TEXT_FONT_PATH, 32)
            self.text_font = ImageFont.load_default()
            self.emoji_font = ImageFont.truetype(EMOJI_FONT_PATH, 32)
        else:  # use cv2
            self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width

    def box_label(
        self,
        box,
        label="",
        color=(128, 128, 128),
        txt_color=(255, 255, 255),
        draw_box=True,
    ):
        """
        @param box: list of 4 values, where the first two are the x,y coordinate of the top left
        of the box, the last two are the x,y coordinate of the bottom left of the box.
        @param label: the text to be added.
        @param color
        @param txt_color
        """
        # Add one xyxy box to image with label
        if self.pil or not is_ascii(label):
            if draw_box:
                self.draw.rectangle(box, width=self.lw, outline=color)  # box
            if label:
                w, h = self.text_font.getsize(label)  # text width, height
                outside = box[1] - h >= 0  # label fits outside box
                if color:
                    self.draw.rectangle(
                        (
                            box[0],
                            box[1] - h if outside else box[1],
                            box[0] + w + 1,
                            box[1] + 1 if outside else box[1] + h + 1,
                        ),
                        fill=color,
                    )

                x = box[0]
                y = box[1] - h if outside else box[1]
                # label_parts = split_text_emoji(label)
                self.draw.text(
                    (x, y),
                    label,
                    font=self.text_font,
                    fill="white",
                    # embedded_color=True,
                )
                # for part in label_parts:
                #     if any(char.isdigit() for char in part) or part.isalpha() or part.isspace():
                #         self.draw.text(
                #             (x, y),
                #             part,
                #             font=self.text_font,
                #             fill="white",
                #             # embedded_color=True,
                #         )
                #         x += self.text_font.getsize(part)[0]
                #     else:
                #         self.draw.text(
                #             (x, y),
                #             part,
                #             font=self.text_font,
                #             fill="white",
                #             # embedded_color=True,
                #         )
                #         x += self.emoji_font.getsize(part)[0] - 5

        else:  # cv2
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            if draw_box:
                cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                tf = max(self.lw - 2, 1)  # font thickness
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 5, thickness=tf)[
                    0
                ]  # text width, height
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(
                    self.im,
                    label,
                    (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    self.lw / 5,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA,
                )

    def update_onscreen_charts(
        self, fighters, stats, show_tracked_chart=True, show_history_charts=False
    ):
        # Turn off interactive mode
        plt.ioff()
        matplotlib.use("Agg")

        self.im = np.array(self.im)
        size = 60
        for fighter in fighters:
            if not show_tracked_chart:
                continue

            (ulx, uly, _, _) = fighter.crop.xyxy_pixels(self.input_width, self.input_height)

            chart, pie_chart_image = move_pie_chart(fighter, stats, size)
            if not chart:
                # Some moves we don't show a chart for.
                continue

            paste_on_top(pie_chart_image, self.im, ulx - 70, uly - 45)

        for fighter in fighters:
            if not show_history_charts:
                continue

            charts, pie_chart_images = move_pie_chart_history(fighter, stats, size)

    def update_offscreen_charts(self, fighters, stats):
        """
        Updates the pie charts on the side of the fighters.
        """
        self.maybe_pad_image()

        index_to_side_x = {
            0: 0,
            1: self.left_padding + self.input_width,
        }
        index_to_bottom_x = {
            0: 0,
            1: self.output_width // 2,
        }
        for fighter in fighters:
            x = index_to_side_x[fighter.fighter_id]
            y = 0
            if True:
                timeline_height = 120

                self.im[y : y + timeline_height, x : x + 400, :3] = disadvantage_ledge_history(
                    fighter=fighter, stats=stats
                )
                y += timeline_height

                self.im[y : y + timeline_height, x : x + 400, :3] = disadvantage_tech_history(
                    fighter=fighter, stats=stats
                )
                y += timeline_height

            if False:
                y = 0
                chart, image = disadvantage_tech_option_chart(fighter, stats, width=400, height=360)
                self.im[y : y + 360, x : x + 400, :] = image

            if False:
                y = 0
                chart, image = disadvantage_ledge_option_chart(
                    fighter, stats, width=400, height=360
                )
                self.im[y : y + 360, x : x + 400, :] = image

            # Default bar charts
            if True:
                im_height = 480
                im_width = 400
                chart, image = move_damage_graph(fighter, stats, width=im_width, height=im_height)
                self.im[y : y + im_height, x : x + im_width, :] = image

            # Chart at bottom of the screen.
            if True:
                x = index_to_bottom_x[fighter.fighter_id]
                y = self.input_height
                im_height = self.bottom_padding
                im_width = self.output_width // 2
                chart, image = move_success_punished_missed_bar_graph(
                    fighter, stats, height=400, width=im_width
                )
                self.im[y : y + im_height, x : x + im_width, :] = image
                pass

    def basic_counter(self, x, fighter, stats):
        """
        Writes a basic counter information on one side of the screen.
        """
        self.box_label(
            (x, 0, x + 20, 40),
            fighter.fighter_name,
            draw_box=False,
            color=None,
            txt_color=(255, 255, 255),
        )
        y = 70
        for key, value in stats.stats[fighter.fighter_id].action_count.items():
            self.box_label(
                (x, y, x + 20, y + 40),
                f"{key}: {value}",
                draw_box=False,
                color=None,
                txt_color=(255, 255, 255),
            )
            y += 30

    def maybe_pad_image(self):
        # Turn off interactive mode
        plt.ioff()
        matplotlib.use("Agg")

        # This is where we add black padding.
        self.im = np.array(self.im)
        if self.im.shape[0] != self.output_height or self.im.shape[1] != self.output_width:
            self.im = np.pad(
                self.im,
                ((0, self.bottom_padding), (self.left_padding, self.right_padding), (0, 0)),
            )

        self.pil = False

    def post_game_summaries(self, fighters, stats):
        """
        Writes a basic counter information on one side of the screen.
        """
        self.maybe_pad_image()
        index_to_x = {
            0: 0,
            1: self.output_width // 2,
        }

        graphs = [
            move_success_punished_missed_bar_graph,
            move_damage_graph,
            defensive_option_chart,
            disadvantage_tech_option_chart,
            disadvantage_ledge_option_chart,
        ]

        for graph in graphs:
            for fighter in fighters:
                x = index_to_x[fighter.fighter_id]
                y = 0
                width = self.output_width // 2
                height = self.output_height

                chart, image = graph(fighter, stats, width=width, height=height)
                self.im[y : y + height, x : x + width, :] = image

            self.write_num_seconds(3)

    def write_num_seconds(self, num_seconds):
        for i in range(num_seconds * 60):
            self.write()

    def result(self):
        # Return annotated image as array
        return np.asarray(self.im)

    def write(self):
        """
        Actually adds a single frame to the output video.
        """
        result = self.result()
        assert result.shape[0] == self.output_height and result.shape[1] == self.output_width, (
            f"Incorrect frame size in Annotator, expected {self.output_width}x{self.output_height} "
            + f"but got {result.shape[1]}x{result.shape[0]}"
        )
        result = cv2.cvtColor(result, cv2.COLOR_RGBA2BGR)
        self.video_writer.write(result)
