import numpy as np
import math
from functools import lru_cache

from bokeh.plotting import figure
from bokeh.transform import cumsum
import pandas as pd

from playaid.graphs.utils import graph_to_image

PIE_CHART_IGNORED_MOVES = [
    "Landing",
    "Walk",
    "Run",
    "Turn",
    "Wait",
    "Jump",
    "ShortHop",
    "Dash",
    "Shield",
    "ShieldDrop",
    "Fall",
    "PlatformDrop",
    "Undefined",
    "Damaged",
]


def make_white_transparent(img):
    # Change all white (also shades of whites) pixels to be transparent
    white = np.all(img[:, :, :3] > 200, axis=2)
    img[white] = 0

    return img


@lru_cache(maxsize=20)
def _move_pie_chart(success, punished, missed, size):
    x = {
        "success": success,
        "punished": punished,
        "missed": missed,
    }

    data = pd.Series(x).reset_index(name="value").rename(columns={"index": "country"})
    data["angle"] = data["value"] / data["value"].sum() * 2 * math.pi
    data["color"] = ["blue", "red", "gray"]

    p = figure(
        outer_width=size,
        outer_height=size,
        width=size,
        height=size,
        title=None,
        toolbar_location=None,
        x_range=(-0.5, 1.0),
        min_border=0,
    )

    p.wedge(
        x=0.25,
        y=1,
        radius=0.7,
        start_angle=cumsum("angle", include_zero=True),
        end_angle=cumsum("angle"),
        line_color="white",
        fill_color="color",
        source=data,
    )

    p.axis.axis_label = None
    p.axis.visible = False
    p.grid.grid_line_color = None
    # Make the background color transparent
    p.background_fill_color = None
    p.border_fill_color = None

    return p, graph_to_image(p, width=size, height=size)


# after a move has been performed, it will gradually stay on and fade away for either this many
# frames or another move has been performed that is pie chart worthy.
MAX_VISIBILITY_FRAMES = 60


def move_pie_chart(fighter, stats, size):
    # Go through the history, if there's one the is not in the ignore move list, then show it
    # but with decreased transparency.
    for frame, history in reversed(stats.stats[fighter.fighter_id].action_timeline.items()):
        if not history.action or history.action in PIE_CHART_IGNORED_MOVES:
            continue

        opacity = 255
        position_in_world = fighter.position_in_world
        if history.end_frame:
            # If we have an end_frame, we're working with a previous action.
            frame_diff = fighter.frame_num - history.end_frame
            # if frame_diff > 30:
            #     break
            opacity = max(
                int(255 * ((MAX_VISIBILITY_FRAMES - frame_diff) / MAX_VISIBILITY_FRAMES)), 0
            )
            position_in_world = history.ending_position_in_world

        if not opacity:
            break

        # Otherwise, return a fade version.
        (success, punished, missed, total) = stats.move_counters(fighter, history.action)

        chart, pie_chart_image = _move_pie_chart(success, punished, missed, size)
        pie_chart_image = make_white_transparent(pie_chart_image)
        mask = pie_chart_image[:, :, 3] > 0
        # Apply the new opacity only to those pixels
        pie_chart_image[mask, 3] = opacity
        return chart, pie_chart_image

    return None, None


def move_pie_chart_history(fighter, stats, size):
    images = []
    for frame, history in reversed(stats.stats[fighter.fighter_id].action_timeline.items()):
        if not history.action or history.action in PIE_CHART_IGNORED_MOVES:
            continue

        # Otherwise, return a fade version.
        (success, punished, missed, total) = stats.move_counters(fighter, history.action)

        chart, pie_chart_image = _move_pie_chart(success, punished, missed, size)
        
        pie_chart_image = make_white_transparent(pie_chart_image)
        return chart, pie_chart_image

    return None, None