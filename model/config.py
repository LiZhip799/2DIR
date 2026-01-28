import os

CONFIG = {
    "SEED": 42,
    "MODEL_NAME": "maxvit_tiny_tf_224.in1k",
    "BATCH_SIZE": 128,
    "NUM_WORKERS": 8,
    "EPOCHS": 50,
    "WARMUP_EPOCHS": 5,
    "LR_BACKBONE": 1e-4,
    "LR_HEAD": 1e-3,
    "WEIGHT_DECAY": 0.05,
    "MAX_GRAD_NORM": 5.0,
    
    # Common Image Settings
    "IMG_SIZE": 224,
    "IMAGENET_MEAN": [0.485, 0.456, 0.406],
    "IMAGENET_STD": [0.229, 0.224, 0.225],

    # Task 1: Matrix / Structure
    "MATRIX": {
        "TRAIN_IMG": "../../data/train/2DIR",
        "TRAIN_CSV": "../../data/train/contact",
        "TEST_IMG": "../../data/test/2DIR",
        "TEST_CSV": "../../data/test/contact",
        "OUTPUT_SIZE": 100,
        "OUTPUT_DIR": "results_matrix"
    },

    # Task 2: Physicochemical Properties
    "PROPS": {
        "TRAIN_IMG": "../../data/train/2DIR",
        "TRAIN_CSV": "../../data/train/train.csv",
        "TEST_IMG": "../../data/test/2DIR",
        "TEST_CSV": "../../data/test/test.csv",
        "SS_COLS": ["helix", "strand", "coil"],
        "PROP_TASKS": ["Rg", "Hbonds", "Buried_Fraction"],
        "STATS_FILE": "norm_stats.json",
        "OUTPUT_DIR": "results_props"
    }
}