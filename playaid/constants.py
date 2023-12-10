import os

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EXPERIMENT_OUTPUT = os.path.join(REPO_ROOT, "experiment_output")
TRACKER_INFERENCE_DATASET_DIR = os.path.join(EXPERIMENT_OUTPUT, "tracker-inference-dataset")
YOLO_DIR = os.path.join(REPO_ROOT, "third_party/yolov5")
XMEM_DIR = os.path.join(REPO_ROOT, "third_party/XMem")
ACTION_RECOG_OUTPUT_DIR = os.path.join(REPO_ROOT, "logs", "action_recog")
SAVED_MODELS = os.path.join(REPO_ROOT, "models")
SAVED_YOLO_MODELS = os.path.join(SAVED_MODELS, "yolo")
SAVED_ACTION_MODELS = os.path.join(SAVED_MODELS, "action")

PLAYAID_ROOT = os.path.join(REPO_ROOT, "playaid")
GAME_DATA_DIR = os.path.join(PLAYAID_ROOT, "game_data")
FRAME_DATA_DIR = os.path.join(GAME_DATA_DIR, "frame_data")
PARAMS_LABELS = os.path.join(GAME_DATA_DIR, "params_labels.csv")
FINETUNE_FILE = os.path.join(PLAYAID_ROOT, "finetuning-mini.jsonl")

TEXT_FONT_PATH = "/Library/Fonts/Arial.ttf"
EMOJI_FONT_PATH = "/System/Library/Fonts/Apple Color Emoji.ttc"
ULT_DATASET_DIR = os.path.realpath(os.path.join(REPO_ROOT, "ult_dataset"))
REPLAYS_DIR = os.path.realpath(os.path.join(ULT_DATASET_DIR, "replays"))
AI_CACHE = os.path.join(REPO_ROOT, "ai_cache")

GROUND_TRUTH_DIR = os.path.realpath(os.path.join(ULT_DATASET_DIR, "ground_truth"))
GROUND_TRUTH_TRAIN = os.path.join(GROUND_TRUTH_DIR, "train.csv")
GROUND_TRUTH_VAL = os.path.join(GROUND_TRUTH_DIR, "val.csv")
GROUND_TRUTH_TEST = os.path.join(GROUND_TRUTH_DIR, "test.csv")
GROUND_TRUTH_EXTRAS = os.path.join(GROUND_TRUTH_DIR, "extras.csv")

GROUND_TRUTH_CHAR_DETECTION_DIR = os.path.join(ULT_DATASET_DIR, "gt_char_detection")

ACTION_GROUND_TRUTH_DIR = os.path.realpath(os.path.join(ULT_DATASET_DIR, "gt_action_detection"))
ACTION_GROUND_TRUTH_TRAIN = os.path.realpath(os.path.join(ACTION_GROUND_TRUTH_DIR, "train"))
ACTION_GROUND_TRUTH_VAL = os.path.realpath(os.path.join(ACTION_GROUND_TRUTH_DIR, "validation"))
ACTION_GROUND_TRUTH_TEST = os.path.realpath(os.path.join(ACTION_GROUND_TRUTH_DIR, "test"))

ULT_DATASET_RAW_CHAR_DIR = os.path.join(ULT_DATASET_DIR, "char_detect_data", "raw")
ULT_DATASET_CLEAN_CHAR_DIR = os.path.join(ULT_DATASET_DIR, "char_detect_data", "clean")
ULT_STAGES_DIR = os.path.join(ULT_DATASET_DIR, "ultimate_stages")
COMPOSITES_DIR = os.path.join(ULT_DATASET_DIR, "composites")

GROUND_TRUTH_VIDEO = os.path.join(ULT_DATASET_DIR, "ult_videos/tweek-mkleo-clip.mp4")
GROUND_TRUTH_SAMPLE = os.path.join(REPO_ROOT, "playaid", "tweek-mkleo-clip-label.csv")
GROUND_TRUTH_SAMPLE_2 = os.path.join(REPO_ROOT, "playaid", "tweek-mkleo-clip-label_xmem_pos.csv")

SYNTH_ACTION_RECOGNITON_DIR = os.path.join(ULT_DATASET_DIR, "synth_char_action_recognition")
SYNTH_ACTION_RECOGNITON_FRAMES_DIR = os.path.join(SYNTH_ACTION_RECOGNITON_DIR, "frames")
SYNTH_ACTION_RECOGNITON_ANNOTATIONS_DIR = os.path.join(SYNTH_ACTION_RECOGNITON_DIR, "annotations")

CHAR_LIST = ["Byleth", "Diddy Kong", "Pikachu", "Joker", "Donkey Kong", "Jigglypuff"]

ACTION_RECOG_NUM_FRAMES_PER_SAMPLE = 4
ACTION_RECOG_FRAME_DELTA = 1
