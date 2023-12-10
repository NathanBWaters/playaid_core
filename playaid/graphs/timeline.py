import re
from PIL import ImageDraw, Image, ImageFont
import numpy as np

from playaid.constants import TEXT_FONT_PATH


def _split_camel_case(s):
    return re.findall(r"[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))", s)


def _split_by_capitalized_word(text):
    return re.split(r"(\b[A-Z][a-z]*\b)", text)


def _timeline(title, fighter, stats, moves, removed_words=[]):
    total = [stats.stats[fighter.fighter_id].action_count[move] or 0 for move in moves]

    tech_history = []
    for frame, history in stats.stats[fighter.fighter_id].action_timeline.items():
        if history.action in moves:
            (success, punished, missed, total) = stats.move_counters(fighter, history.action)
            tech_history.append((history.action, frame, (success, punished, missed, total)))

    # Define the size of the image and the sections
    title_height = 20
    image_width = 400
    image_height = 120
    num_sections = 5
    section_width = image_width // num_sections
    border_size = 2

    # Create the image
    image = Image.new("RGB", (image_width, image_height), "white")
    draw = ImageDraw.Draw(image)

    # Add black borders between sections and around the whole image
    for i in range(num_sections):
        if i != 0:
            # Draw vertical lines for section borders
            draw.line(
                [(i * section_width, title_height), (i * section_width, image_height)],
                fill="black",
                width=border_size,
            )
        if i == num_sections - 1:
            # Draw a border around the whole image
            draw.rectangle(
                [(0, 0), (image_width - border_size, image_height - border_size)],
                outline="black",
                width=border_size,
            )
            # Draw a smaller border to separate the title section.
            draw.rectangle(
                [
                    (0, title_height),
                    (image_width - border_size, image_height - border_size),
                ],
                outline="black",
                width=border_size,
            )

    # Draw the text
    fnt = ImageFont.truetype(TEXT_FONT_PATH, 15)  # Adjust font path and size as needed
    draw.text((3, 2), title, font=fnt, fill=(0, 0, 0))

    for i in range(num_sections):
        if i >= len(tech_history):
            break

        text, frame_num, (success, punished, missed, total) = tech_history[
            len(tech_history) - i - 1
        ]
        text_width, text_height = draw.textsize(text, font=fnt)
        x = i * section_width + 3

        # Draw the frame number
        draw.text((x, title_height), "#" + str(i + 1), font=fnt, fill=(0, 0, 0))

        # Draw the success count
        # BUG: success + missed
        draw.text((x, image_height - 20), str(success + missed), font=fnt, fill=(255, 0, 0))
        # Draw the punished count
        draw.text(
            (x + section_width - 17, image_height - 20),
            str(punished),
            font=fnt,
            fill=(0, 0, 255),
        )

        for word in removed_words:
            text = text.replace(word, "")
        text_split = _split_camel_case(text)

        starting_y = (image_height - text_height) // 2
        for j in range(len(text_split)):
            y = starting_y + (j * 20)
            draw.text((x + (section_width // 3), y), text_split[j], font=fnt, fill=(0, 0, 0))

    # BUG: No clue why I need to switch B and R.
    output = np.array(image)[:, :, ::-1]
    return output


def disadvantage_tech_history(fighter, stats):
    moves = set(
        [
            "TechInPlace",
            "TechRoll",
            "NormalGetUp",
            "GetUpAttack",
            "DownWait",
            "MissedTech",
        ]
    )

    return _timeline(
        f"{fighter.fighter_name.capitalize()} Disadvantage Tech History",
        fighter,
        stats,
        moves,
    )


def disadvantage_ledge_history(fighter, stats):
    moves = set(
        [
            "LedgeAttack",
            "LedgeNormalGetUp",
            "LedgeRoll",
            "LedgeJump",
        ]
    )

    return _timeline(
        f"{fighter.fighter_name.capitalize()} Disadvantage Ledge History",
        fighter,
        stats,
        moves,
        ["Ledge"],
    )
