"""
Helps visualize the data being fed to the models
"""
import streamlit as st
import torch
import random
from statistics import mean
import torch.nn.functional as F

import os
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

from playaid.ult_action_dataset import UltActionRecogDataset
from playaid.anim_ontology import MOVE_TO_CLASS_ID, TRAINED_ACTIONS_2_17, ONTOLOGY
from playaid.models.rnn_action_detector import RNNActionDetector
from playaid.constants import CHAR_LIST, GROUND_TRUTH_SAMPLE_2, SAVED_MODELS

# from playaid.annotator import get_img_from_fig

torch.set_printoptions(precision=3, sci_mode=False)


# Build confusion matrix
def confusion_matrix_image(y_true, y_pred, classes):
    """
    Takes in two 1d np arrays of integers and makes an image from it.
    """
    cf_matrix = confusion_matrix(
        y_true, y_pred, labels=[i for i in range(len(actions))], normalize="true"
    )
    df_cm = pd.DataFrame(
        cf_matrix,
        index=[i for i in classes],
        columns=[i for i in classes],
    )
    df_cm.index.name = "Actual"
    df_cm.columns.name = "Predicted"
    plt.figure(figsize=(12, 7))
    map = sn.heatmap(df_cm, annot=True)
    return get_img_from_fig(map.figure)


def vis_animations(
    parent,
    split,
    actions=TRAINED_ACTIONS_2_17,
    img_dimensions=128,
    num_frames_per_sample=3,
    frame_delta=1,
    randomize_stage_background=True,
    move_stage_background=True,
    total=3,
    char_subset=CHAR_LIST,
    ground_truth_offset=0,
    synth_difficulty=0,
):
    """
    Renders synthetic letters
    """
    loader = UltActionRecogDataset(
        split=split,
        num_samples=total,
        img_dimension=img_dimensions,
        num_frames_per_sample=num_frames_per_sample,
        frame_delta=frame_delta,
        anim_subset=actions,
        char_subset=char_subset,
        randomize_stage_background=randomize_stage_background,
        move_stage_background=move_stage_background,
        ground_truth_offset=ground_truth_offset,
        synth_difficulty=synth_difficulty,
        # ground_truth_csv=GROUND_TRUTH_SAMPLE_2,
    )

    model = RNNActionDetector.load_from_checkpoint(
        # "/Users/nathan/repo/smash_ai/logs/action_recog/byleth/version_24/checkpoints/epoch=441-step=14144.ckpt"
        os.path.join(SAVED_MODELS, "best-byleth-action-july-17.ckpt"),
        fighter_name="byleth",
        actions=actions,
    )
    model.eval()

    num_correct = 0
    confidence_list = []

    labels = []
    preds = []

    seq_length = -1
    for i in range(total):
        random.seed(2 * i + 48)
        input_frames, char_label, action_label, data = loader[i]
        seq = input_frames.shape[0]
        seq_length, c, h, w = input_frames.shape
        action_id = action_label.view(seq_length)

        captions = []
        # Output shape [seq_length, num_actions]
        predictions = model(input_frames.unsqueeze(0))
        probabilities = torch.exp(predictions)
        for i in range(seq_length):
            predicted_action_id = int(torch.argmax(predictions[i]))
            predicted_action = model.actions[predicted_action_id]
            confidence = float(probabilities[i][predicted_action_id]) * 100.0

            labels.append(int(action_label[i]))
            preds.append(predicted_action_id)
            # parent.write(f"Probabilities: {probabilities}")
            # parent.write(f"Predictions: {predictions}")
            # Just make sure the strings are the same and not go by model / passed action id because
            # those can be different.
            # is_accurate = actions[action_label[-1]] == predicted_action
            gt_label = actions[int(action_id[i])]
            is_accurate = gt_label == predicted_action
            caption = f"{'✅' if is_accurate else '❌'} "
            caption += f"Pred: {predicted_action} {confidence:.2f}%"
            if not is_accurate:
                caption += f" | GT: {gt_label}"
                caption += f" {data['frame_paths'][i]}"
            captions.append(caption)
            num_correct += is_accurate
            confidence_list.append(confidence)

        parent.image(
            data["frames"],
            caption=captions,
            width=200,
            clamp=True,
        )

        parent.write("-" * 80)
    parent.write(
        f"{total} samples | {seq_length} frames | {frame_delta} delta | "
        + f"random background {randomize_stage_background} | moved stage {move_stage_background}"
    )
    parent.write(f"actions: {actions}")
    parent.write(f"% correct: {(num_correct / float(total * seq_length)):.2f}")
    parent.write(f"mean confidence: {(mean(confidence_list)):.2f}")

    # conf_matrix_image = confusion_matrix_image(
    #     np.array(labels), np.array(preds), actions
    # )
    # parent.image(conf_matrix_image)


if __name__ == "__main__":

    st.set_page_config(layout="wide")

    left, right = st.columns((1, 1))
    # actions = ["ForwardSmash", "NeutralAir", "BackAir", "Dash", "Unknown"]
    actions = list(MOVE_TO_CLASS_ID.keys())
    num_frames = 4
    frame_delta = 5
    total = 10

    left.title("Train")
    vis_animations(
        left,
        "train",
        char_subset=["Byleth"],
        actions=actions,
        total=total,
        num_frames_per_sample=num_frames,
        frame_delta=frame_delta,
        synth_difficulty=1,
    )
    right.title("Validation")
    vis_animations(
        right,
        "validation",
        char_subset=["Byleth"],
        actions=actions,
        total=total,
        num_frames_per_sample=num_frames,
        frame_delta=frame_delta,
        synth_difficulty=0,
    )
    print("Good")
