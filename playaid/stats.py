"""Keeps track of various stats of the game"""

from addict import Dict
import yaml
import os
from tqdm import tqdm
from dictdiffer import diff
from functools import lru_cache

from playaid import constants
from playaid.fighter import Fighter
from playaid.timeline import update_fighters_from_timeline, load_ground_truth_from_path
from playaid.frame_data import FIGHTER_FRAME_DATA


IGNOREABLE_ACTIONS = [
    # "Turn",
    # "Squat",
    # "PlatformDrop",
    # "Landing",
    # "Wait",
    # "Dash",
    "Undefined",
]


@lru_cache(maxsize=2)
def get_stats_at_frame(frame_num: int, video_path: str, label_path: str, log_offset=0):
    timeline = load_ground_truth_from_path(label_path, log_offset=log_offset)
    fighters = [Fighter(frame_num=0, data=json_data) for json_data in timeline[0]]

    stats = Stats(video_path)
    for i in tqdm(range(frame_num)):
        if i >= len(timeline):
            break
        update_fighters_from_timeline(i, timeline[i], fighters)
        stats.record_frame(fighters)
    return stats


def frame_subset_from_dict(dict, start_frame, end_frame):
    """
    Given a dictionary where the keys are frame numbers, return a list of those frames.
    """
    valid_frames = []
    for frame_num in dict.keys():
        if frame_num < start_frame:
            continue

        if frame_num > end_frame:
            break

        valid_frames.append(frame_num)

    return valid_frames


class Stats:
    def __init__(self, input_video_path):
        self.input_video_path = input_video_path
        self.src_folder, self.file_name = os.path.split(self.input_video_path)
        self.video_name, _ = os.path.splitext(self.file_name)
        parent_folder = os.path.basename(self.src_folder)
        self.exp_name = os.path.join(parent_folder, self.video_name)
        self.output_dir = os.path.join(constants.AI_CACHE, self.exp_name, "stats")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.stats = Dict()

    def record_frame(self, fighters):
        """ """
        self.update_fighter(fighters[0], fighters[1])
        self.update_fighter(fighters[1], fighters[0])

        self.post_update(fighters)

    def update_fighter(self, fighter, other_fighter):
        if fighter.damage_delta:
            # Record that the move was punished
            self.stats[fighter.fighter_id].punished_action_count[
                fighter.previous_non_damaged_action
            ] += 1

            # Record how much damage the fighter took for the move.
            self.stats[fighter.fighter_id].punished_action_damage[
                fighter.previous_non_damaged_action
            ] += fighter.damage_delta

            # Record which move punished this move
            self.stats[fighter.fighter_id].punished_action_by_action_count[
                fighter.previous_non_damaged_action
            ][other_fighter.action] += 1

            # Records when a move of yours was hit by your opponent at what time and for how much.
            self.stats[fighter.fighter_id].punish_history[fighter.previous_non_damaged_action][
                fighter.frame_num
            ] = {
                "action": other_fighter.action,
                "damage_delta": fighter.damage_delta,
                "frame_number": fighter.frame_num,
            }

        # Bug: if there's a legitimate case of using the same move twice in a row then it will
        # only count once.
        if fighter.new_action:
            self.stats[fighter.fighter_id].action_count[fighter.action] += 1

            # Record when it happened in order, with the key being the frame and the value being
            # the action.
            action_timeline = self.stats[fighter.fighter_id].action_timeline

            # If there was a previous action, record when it ended.
            if action_timeline.keys():
                last_action = action_timeline[list(action_timeline.keys())[-1]]
                last_action.end_frame = fighter.frame_num, -1
                last_action.ending_position_in_world = fighter.position_in_world

            timeline_data = action_timeline[fighter.frame_num]
            timeline_data.action = fighter.action
            timeline_data.starting_position_in_world = fighter.position_in_world
            timeline_data.start_frame = fighter.frame_num

        # Count whether a move was successful.
        if other_fighter.damage_delta:
            # Only mark a move as having been successful once, we don't want to count multihits
            # multiple times.
            if not fighter.previous_attack_connected:
                self.stats[fighter.fighter_id].successful_action_count[fighter.action] += 1

            # Record how much damage the fighter did with the move.
            self.stats[fighter.fighter_id].successful_action_damage[
                fighter.action
            ] += other_fighter.damage_delta

            # Records when a move of yours successfully hit the opponent at what time and for how
            # much.
            self.stats[fighter.fighter_id].success_history[fighter.previous_non_damaged_action][
                fighter.frame_num
            ] = {
                "action": other_fighter.previous_non_damaged_action,
                "damage_delta": other_fighter.damage_delta,
                "frame_number": fighter.frame_num,
            }

    def post_update(self, fighters):
        for fighter in fighters:
            if self.stats[fighter.fighter_id].latest_action != fighter.action:
                self.stats[fighter.fighter_id].latest_action_frame = fighter.frame_num
                self.stats[fighter.fighter_id].latest_action = fighter.action

        for fighter, other_fighter in [[fighters[0], fighters[1]], [fighters[1], fighters[0]]]:
            #################
            #  SHIELD STUN  #
            #################
            if (
                other_fighter.new_action
                and other_fighter.action == "ShieldStun"
                # This is a hack to prevent situations where projectiles are the cause.
                and fighter.using_damage_move
            ):
                self.stats.history[fighter.frame_num] = (
                    f"{fighter.fighter_name} hit {other_fighter.fighter_name}'s shield with "
                    + f"{fighter.action}, putting {other_fighter.fighter_name} into ShieldStun"
                )
                # No need to add anything else
                continue

            #################
            #    DAMAGE     #
            #################
            elif other_fighter.damage_delta:
                self.stats.history[fighter.frame_num] = (
                    f"{fighter.fighter_name}, who is at {fighter.damage:.2f} damage, used "
                    + f"{fighter.action} to punish {other_fighter.fighter_name} use of "
                    + f"{other_fighter.previous_action} for {other_fighter.damage_delta:.2f} damage"
                )

            #################
            #     ACTION    #
            #################
            elif (
                fighter.new_action
                # Because we're comparing A to B and B to A, make sure we're not adding duplicate
                # information.
                and not self.stats.history[fighter.frame_num]
                and fighter.action not in IGNOREABLE_ACTIONS
            ):
                # print(f'New action #{fighter.frame_num} {fighter.fighter_name}: {fighter.previous_action} -> {fighter.action}')
                if fighter.previous_action:
                    # self.stats.history[fighter.frame_num - 1] = (
                    #     f"P{fighter.fighter_id + 1} {fighter.fighter_name}@{fighter.damage:.2f} finished "
                    #     + f"{fighter.previous_action}#{self.stats[fighter.fighter_id].action_count[fighter.previous_action]} "
                    # )
                    self.stats.history[fighter.frame_num] = self.to_sentence(
                        fighter, other_fighter, f"ended move {fighter.action}"
                    )
                    # if not fighter.previous_attack_connected:
                    #     self.stats.history[-1] += '- it did not land'
                self.stats.history[fighter.frame_num] = self.to_sentence(
                    fighter, other_fighter, f"started move {fighter.action}"
                )

            if fighter.previous_damage and not fighter.damage:
                self.stats.history[
                    fighter.frame_num
                ] = f"P{fighter.fighter_id + 1} {fighter.fighter_name} died"

    def to_sentence(self, fighter, other_fighter, specific_string):
        text = f"""
        P{fighter.fighter_id + 1} {fighter.fighter_name} {specific_string} at position
        {fighter.pos_x:.2f}x,{fighter.pos_y:.2f}y with {fighter.damage:.2f}% damage.
        Opponent P{other_fighter.fighter_id + 1} {other_fighter.fighter_name} is at frame
        {other_fighter.animation_frame_num} of move {other_fighter.action} and is
        {other_fighter.offset_str(fighter)} from {fighter.fighter_name} with
        {other_fighter.damage:.2f}% damage.
        """
        return " ".join(text.split())

    def move_counters(self, fighter, move):
        total = self.stats[fighter.fighter_id].action_count[move] or 0
        success = self.stats[fighter.fighter_id].successful_action_count[move] or 0
        punished = self.stats[fighter.fighter_id].punished_action_count[move] or 0
        missed = max(total - success - punished, 0)
        return (success, punished, missed, total)

    def move_counter_str(self, fighter, move):
        (success, punished, missed, total) = self.move_counters(fighter, move)
        return f"{success}✅, {punished}❌, {missed}⭕️, {total}"

    def move_set(self, fighter, start_frame, end_frame):
        """
        Returns all the moves that were performed between the start-end frame.
        """
        action_timeline = self.stats[fighter.fighter_id].action_timeline

        moves = []
        frame_subset = frame_subset_from_dict(action_timeline, start_frame, end_frame)
        for frame_num in frame_subset:
            timeline_data = action_timeline[frame_num]
            moves.append(timeline_data.action)

        return list(set(moves))

    def damage_causing_move_set(self, fighter, start_frame, end_frame):
        """
        Returns all the moves that do damage and were performed between the start-end frame.
        """
        move_set = self.move_set(fighter, start_frame, end_frame)

        return [move for move in move_set if move in FIGHTER_FRAME_DATA[fighter.fighter_name]]

    def frame_data_str(self, fighter, start_frame, end_frame):
        moves = self.damage_causing_move_set(fighter, start_frame, end_frame)
        frame_data = {}

        for move in moves:
            frame_data[move] = FIGHTER_FRAME_DATA[fighter.fighter_name][move]

        return "\n".join([f"{move} - {str(data)}" for move, data in frame_data.items()])

    def counter_summaries_str(self, fighter, start_frame, end_frame):
        moves = self.damage_causing_move_set(fighter, start_frame, end_frame)
        counter_data = []
        for move in moves:
            (success, punished, missed, total) = self.move_counters(fighter, move)
            counter_data.append(
                f"{move} has landed successfully {success} times, punished {punished} times, and "
                + f"whiffed {missed} times"
            )

        return "\n".join(counter_data)

    def granular_history(self, fighter, moves, history):
        """
        Extracts a history out of the punish/success history and converts it into a string.
        """
        str_history = []
        for move in moves:
            if not history[move]:
                # Skip it.
                continue

            move_history_str = f"{move}:\n"
            for _, move_history in history[move].items():
                move_history_str += f"- {str(move_history)}\n"
            str_history.append(move_history_str)

        return "\n".join(str_history)

    def punish_history(self, fighter, start_frame, end_frame):
        moves = self.damage_causing_move_set(fighter, start_frame, end_frame)
        return self.granular_history(fighter, moves, self.stats[fighter.fighter_id].punish_history)

    def success_history(self, fighter, start_frame, end_frame):
        moves = self.damage_causing_move_set(fighter, start_frame, end_frame)
        return self.granular_history(fighter, moves, self.stats[fighter.fighter_id].success_history)

    def history_subset(self, start_frame, end_frame):
        """
        Returns all the moves there were performed between the start-end frame.
        """
        history = []
        frame_subset = frame_subset_from_dict(self.stats.history, start_frame, end_frame)
        for frame_num in frame_subset:
            history.append((frame_num, self.stats.history[frame_num]))

        return history

    def stats_path(self, frame_num: int):
        return os.path.join(self.output_dir, f"stats_{frame_num}.yaml")

    def write_all_stats(self, timeline, fighters, interval=1):
        """
        Writes all stats to ai_cache/video/stats/stats_<frame_num>.yaml
        """
        if os.path.exists(self.stats_path(0)):
            print("Already created stats files")
            return

        print(f"Creatings stats files at {self.output_dir}")
        for i in tqdm(range(len(timeline))):
            update_fighters_from_timeline(i, timeline[i], fighters)
            self.record_frame(fighters)
            if i % interval != 0:
                continue

            path = self.stats_path(i)
            with open(path, "w") as f:
                yaml.dump(self.stats.to_dict(), f)
        print(f"Finished creating stats files at {self.output_dir}")

    def get_stats(self, frame_num: int):
        path = self.stats_path(frame_num)
        if not os.path.exists(path):
            return False, {}

        with open(path, "r") as f:
            try:
                stats = Dict(yaml.safe_load(f))
                return True, stats
            except Exception:
                return False, {}

    def load_stats(self, frame_num: int):
        res, self.stats = self.get_stats(frame_num)
        return res

    def stat_diff(self, start_frame: int, end_frame: int):
        res, start_stat = self.get_stats(start_frame)
        res, end_stat = self.get_stats(end_frame)
        diff_results = diff(start_stat, end_stat)
        return res and res, diff_results, start_stat, end_stat

    def instances_of_hits_on_shield(self):
        """
        Returns history instances of hits on shield
        """
        return self.instances_of("into ShieldStun")

    def instances_of(self, instance_key, offset=13):
        instances = []
        timestamps = list(self.stats.history.keys())
        for i, key in enumerate(timestamps):
            history = self.stats.history[key]
            if instance_key in history:
                instance = []
                for j in range(max(0, i - offset), min(i + offset, len(timestamps))):
                    timestamp = timestamps[j]
                    instance.append((timestamp, self.stats.history[timestamp]))
                instances.append(instance)
        return instances
