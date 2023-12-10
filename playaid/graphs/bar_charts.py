from functools import lru_cache
import math
from bokeh.models import BasicTickFormatter
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.transform import factor_cmap
from bokeh.palettes import TolRainbow20

from playaid.graphs.utils import graph_to_image

SYMBOL_TO_WORD = {
    "F": "Forward",
    "D": "Down",
    "B": "Back",
    "U": "Up",
    "N": "Neutral",
    "Z": "Z",
}

IGNORE_GROUP = [
    "Movement",
    "Defensive",
]
ANIM_TO_CATEGORY = {
    "Jump": ["Jump", "Landing", "Fall", "ShortHop"],
    # idk if Roll should be here.
    "Grnd": ["Wait", "Squat", "Turn", "Roll"],
    "Dash": ["DashAttack"],
    # "AirD": ["AirDodge"],
    # "Shld": ["Shield"],
}


def bar_graph(actions, counts, width=400, height=360, title="actions", orientation=0):
    source = ColumnDataSource(data=dict(actions=actions, counts=counts))

    p = figure(
        x_range=FactorRange(*actions),
        height=360,
        width=400,
        toolbar_location=None,
        title=title,
    )
    p.vbar(
        x="actions",
        top="counts",
        width=0.9,
        source=source,
        line_color="white",
        fill_color=factor_cmap("actions", palette=TolRainbow20, factors=actions),
    )

    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.y_range.end = max(counts)
    # p.xaxis.major_label_orientation = "vertical"
    if orientation:
        p.xaxis.major_label_orientation = orientation

    p.yaxis[0].ticker.desired_num_ticks = max(counts) + 1
    p.yaxis.formatter = BasicTickFormatter(use_scientific=False)

    return p


def data_to_ys(data, symbol_to_word, anim_to_category, moves):
    """
    TODO
    """
    ys = []
    for move in moves:
        key = ""
        if move[0] == "?":
            total = sum(data.values())
            accounted = sum(ys)
            ys.append(total - accounted)
        elif move[0] in IGNORE_GROUP:
            ys.append(data[move[1]] or 0)
        elif move[0] in anim_to_category:
            value = sum([data[key] or 0 for key in anim_to_category[move[0]]])
            ys.append(value)
        else:
            key = (
                symbol_to_word[move[1]] + move[0]
                if move[1] in symbol_to_word
                else move[1] + move[0]
            )

            value = data[key] or 0
            ys.append(value)

    return ys


@lru_cache(maxsize=2)
def _defensive_option_chart(moves, counts, title, width=400, height=320):
    graph = bar_graph(
        actions=moves,
        counts=counts,
        title=title,
        orientation=math.pi / 4,
    )
    image = graph_to_image(graph, width, height)
    return graph, image


def defensive_option_chart(fighter, stats, width=400, height=320):
    """ """
    moves = [
        ("Movement", "Jump"),
        ("Movement", "ShortHop"),
        ("Movement", "Walk"),
        ("Movement", "Run"),
        ("Movement", "Squat"),
        ("Movement", "Wait"),
        ("Defensive", "Shield"),
        ("Defensive", "SpotDodge"),
        ("Defensive", "Roll"),
        ("Defensive", "AirDodge"),
        ("Defensive", "Parry"),
    ]
    counts = [stats.stats[fighter.fighter_id].action_count[move[1]] or 0 for move in moves]

    chart, image = _defensive_option_chart(
        tuple(moves),
        tuple(counts),
        title=f"{fighter.fighter_name.title()} Defensive Options",
        width=width,
        height=height,
    )

    return chart, image


@lru_cache(maxsize=2)
def _success_vs_punished_graph(
    moves,
    success,
    punished,
    title,
    width=400,
    height=240,
    punished_label="punished",
    success_label="success",
):
    source = ColumnDataSource(
        data=dict(
            actions=moves,
            success=success,
            punished=punished,
        )
    )

    p = figure(
        x_range=FactorRange(*moves),
        height=height,
        width=width,
        toolbar_location=None,
        title=title,
    )
    p.vbar_stack(
        ["punished", "success"],
        x="actions",
        width=0.9,
        color=["red", "blue"],
        source=source,
        legend_label=[punished_label, success_label],
    )

    p.xgrid.grid_line_color = None
    p.y_range.start = 0

    graph_image = graph_to_image(p, width=width, height=height)

    return p, graph_image


def move_damage_graph(fighter, stats, width=400, height=480):
    moves = [
        ("Jab", ""),
        ("Dash", ""),
        ("Tilt", "F"),
        ("Tilt", "U"),
        ("Tilt", "D"),
        ("Smash", "F"),
        ("Smash", "U"),
        ("Smash", "D"),
        ("Special", "N"),
        ("Special", "F"),
        ("Special", "U"),
        ("Special", "D"),
        ("Air", "N"),
        ("Air", "F"),
        ("Air", "B"),
        ("Air", "U"),
        ("Air", "D"),
        ("Air", "Z"),
        ("Grab", ""),
    ]

    success = data_to_ys(
        stats.stats[fighter.fighter_id]["successful_action_damage"],
        SYMBOL_TO_WORD,
        ANIM_TO_CATEGORY,
        moves,
    )

    punished = data_to_ys(
        stats.stats[fighter.fighter_id]["punished_action_damage"],
        SYMBOL_TO_WORD,
        ANIM_TO_CATEGORY,
        moves,
    )

    chart, graph_image = _success_vs_punished_graph(
        tuple(moves),
        tuple(success),
        tuple(punished),
        title=f"{fighter.fighter_name.title()} Sum damage output for move / sum damage received for move",
        width=width,
        height=height,
        punished_label="damaged received",
        success_label="damage output",
    )

    return chart, graph_image


def disadvantage_tech_option_chart(fighter, stats, width=400, height=360):
    moves = [
        ("", "TechInPlace"),
        ("", "TechRoll"),
        ("", "NormalGetUp"),
        ("", "GetUpAttack"),
        ("", "DownWait"),
    ]
    total = [stats.stats[fighter.fighter_id].action_count[move[1]] or 0 for move in moves]

    punished = [
        stats.stats[fighter.fighter_id].punished_action_count[move[1]] or 0 for move in moves
    ]
    successful = [total[i] - punished[i] for i in range(len(total))]

    title = f"{fighter.fighter_name.title()} Disadvantage Tech Options"
    return _success_vs_punished_graph(
        tuple(moves),
        tuple(successful),
        tuple(punished),
        title,
        width=width,
        height=height,
    )


def disadvantage_ledge_option_chart(fighter, stats, width=400, height=360):
    """ """
    moves = [
        ("Disadvantage Ledge Option", "Attack"),
        ("Disadvantage Ledge Option", "NormalGetUp"),
        ("Disadvantage Ledge Option", "Hang"),
        ("Disadvantage Ledge Option", "Roll"),
        ("Disadvantage Ledge Option", "Jump"),
    ]
    total = [stats.stats[fighter.fighter_id].action_count["Ledge" + move[1]] or 0 for move in moves]
    punished = [
        stats.stats[fighter.fighter_id].punished_action_count["Ledge" + move[1]] or 0
        for move in moves
    ]
    successful = [total[i] - punished[i] for i in range(len(total))]

    title = f"{fighter.fighter_name.title()} Disadvantage Ledge Options"
    return _success_vs_punished_graph(
        tuple(moves),
        tuple(successful),
        tuple(punished),
        title,
        width=width,
        height=height,
    )


def _action_chart(moves, counts, title):
    graph = bar_graph(actions=moves, counts=counts, title="Actions")
    graph_image = graph_to_image(graph)
    return graph, graph_image


def action_chart(fighter, stats):
    """ """
    moves = [
        ("Jab", "N"),
        ("Tilt", "F"),
        ("Tilt", "U"),
        ("Tilt", "D"),
        ("Smash", "F"),
        ("Smash", "U"),
        ("Smash", "D"),
        ("Special", "N"),
        ("Special", "F"),
        ("Special", "U"),
        ("Special", "D"),
        ("Throw", "F"),
        ("Throw", "B"),
        ("Throw", "U"),
        ("Throw", "D"),
        ("Air", "N"),
        ("Air", "F"),
        ("Air", "B"),
        ("Air", "U"),
        ("Air", "D"),
        ("Air", "Z"),
    ]
    symbol_to_word = {
        "F": "Forward",
        "D": "Down",
        "B": "Back",
        "U": "Up",
        "N": "Neutral",
        "Z": "Z",
    }
    counts = [
        stats.stats[fighter.fighter_id].action_count[
            symbol_to_word[move[1]] + move[0] if move[0] != "Jab" else "Jab"
        ]
        or 0
        for move in moves
    ]
    title = f"{fighter.fighter_name.title()} Actions"
    graph, graph_image = _action_chart(tuple(moves), tuple(counts), title)

    return graph, graph_image


@lru_cache(maxsize=2)
def _move_success_punished_missed_bar_graph(
    moves,
    success,
    punished,
    missed,
    title,
    width=720,
    height=400,
    orientation=0,
):
    source = ColumnDataSource(
        data=dict(
            actions=moves,
            success=success,
            punished=punished,
            missed=missed,
        )
    )

    p = figure(
        x_range=FactorRange(*moves),
        height=height,
        width=width,
        toolbar_location=None,
        title=title,
    )
    p.vbar_stack(
        ["missed", "punished", "success"],
        x="actions",
        width=0.9,
        color=["gray", "red", "blue"],
        source=source,
        legend_label=["missed", "punished", "success"],
    )

    p.xgrid.grid_line_color = None
    if orientation:
        p.xaxis.major_label_orientation = orientation

    p.y_range.start = 0
    graph_image = graph_to_image(p, height=height, width=width)
    return p, graph_image


def move_success_punished_missed_bar_graph(fighter, stats, width=720, height=400):
    moves = [
        # ("Movement", "Jump"),
        # ("Movement", "ShortHop"),
        # ("Movement", "Walk"),
        # ("Movement", "Run"),
        # ("Movement", "Squat"),
        # ("Movement", "Wait"),
        # ("Defensive", "Shield"),
        # ("Defensive", "SpotDodge"),
        # ("Defensive", "Roll"),
        # ("Defensive", "AirDodge"),
        # ("Defensive", "Parry"),
        ("Jab", ""),
        ("Dash", ""),
        ("Tilt", "F"),
        ("Tilt", "U"),
        ("Tilt", "D"),
        ("Smash", "F"),
        ("Smash", "U"),
        ("Smash", "D"),
        ("Special", "N"),
        ("Special", "F"),
        ("Special", "U"),
        ("Special", "D"),
        ("Air", "N"),
        ("Air", "F"),
        ("Air", "B"),
        ("Air", "U"),
        ("Air", "D"),
        ("Air", "Z"),
        ("Grab", ""),
    ]

    success = data_to_ys(
        stats.stats[fighter.fighter_id]["successful_action_count"],
        SYMBOL_TO_WORD,
        ANIM_TO_CATEGORY,
        moves,
    )

    punished = data_to_ys(
        stats.stats[fighter.fighter_id]["punished_action_count"],
        SYMBOL_TO_WORD,
        ANIM_TO_CATEGORY,
        moves,
    )
    total = data_to_ys(
        stats.stats[fighter.fighter_id]["action_count"],
        SYMBOL_TO_WORD,
        ANIM_TO_CATEGORY,
        moves,
    )

    missed = []
    for i in range(len(moves)):
        miss_count = max(total[i] - success[i] - punished[i], 0)
        missed.append(miss_count)

    title = f"{fighter.fighter_name.title()} Successful / Punished / Missed Count"
    return _move_success_punished_missed_bar_graph(
        tuple(moves),
        tuple(success),
        tuple(punished),
        tuple(missed),
        title,
        width=width,
        height=height,
        orientation=math.pi / 4,
    )
