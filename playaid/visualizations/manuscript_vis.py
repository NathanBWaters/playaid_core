"""
Helps visualize the data being fed to the models
"""
import streamlit as st
import torch
import numpy as np
import os

from playaid.ult_action_dataset import UltActionRecogDataset
from playaid.anim_ontology import MOVE_TO_CLASS_ID
from playaid.models.rnn_action_detector import RNNActionDetector
from playaid.manuscript import Manuscript
from playaid.constants import (
    ULT_DATASET_DIR,
    EXPERIMENT_OUTPUT,
    YOLO_DIR,
    ACTION_RECOG_OUTPUT_DIR,
    ACTION_RECOG_NUM_FRAMES_PER_SAMPLE,
    ACTION_RECOG_FRAME_DELTA,
)

torch.set_printoptions(precision=3, sci_mode=False)


def detect(manuscript, frame_num):
    output = manuscript.detect_actions_for_frame(frame_num)
    byleth_data = output["byleth"]
    print("byleth_data: ", byleth_data)
    byleth_crops = byleth_data["crops"]

    byleth_crops = byleth_crops.squeeze(0)
    byleth_crops = byleth_crops.permute(0, 2, 3, 1)

    caption = [
        f"#{byleth_data['frame_nums'][i]} - {byleth_data['crop_data'][i]['x_pixels']}x, {byleth_data['crop_data'][i]['y_pixels']}y"
        for i in range(len(byleth_data["frame_nums"]))
    ]

    st.image(
        [
            np.array(byleth_crops[i, :, :, :])
            for i in range(ACTION_RECOG_NUM_FRAMES_PER_SAMPLE)
        ],
        caption=caption,
        width=200,
        clamp=True,
    )

    st.write(f"Probabilities: {byleth_data['probabilities']}")
    st.write(
        f"Predicted action: {byleth_data['predicted_action']} + "
        f"({byleth_data['predicted_action_id']}) - {byleth_data['confidence']:.2f}%"
    )
    st.write("-" * 80)


def vis_animations():
    """
    Renders synthetic letters
    """
    st.write("Creating manuscript")
    manuscript = Manuscript()

    # st.text(str(manuscript))

    # 13 should give me 10, 13, 16, which are all of MKLeo squatting.
    detect(manuscript, 13)
    detect(manuscript, 374)
    detect(manuscript, 405)
    detect(manuscript, 481)
    detect(manuscript, 530)
    detect(manuscript, 582)


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    vis_animations()
    print("Good")
