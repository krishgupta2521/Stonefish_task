#!/usr/bin/env python3

from pathlib import Path
import argparse
import sys

import torch
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

DATA_YAML = Path(
    "/home/krish/Documents/buoy_dataset_real_world/data.yaml"
)

MODEL_WEIGHTS = "yolo26n.pt"

OUTPUT_DIRECTORY = (
    "/home/krish/Documents/Development/kyushu_project/runs/detect"
)

EPOCHS = 300
PATIENCE = 50
IMAGE_SIZE = 640
BATCH_SIZE = 8
WORKERS = 4
SEED = 42


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_model(device="auto"):

    print("\n" + "=" * 60)
    print("       UNDERWATER BUOY DETECTION - YOLO TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Verify dataset configuration
    # --------------------------------------------------------

    if not DATA_YAML.is_file():
        sys.exit(
            f"\nERROR: Dataset YAML not found:\n{DATA_YAML}"
        )

    print(f"\nDataset YAML : {DATA_YAML}")
    print(f"Model        : {MODEL_WEIGHTS}")

    # --------------------------------------------------------
    # 2. Check GPU availability
    # --------------------------------------------------------

    cuda_available = torch.cuda.is_available()

    if device == "0" and not cuda_available:
        sys.exit(
            "\nERROR: CUDA is unavailable.\n"
            "Your NVIDIA driver or PyTorch CUDA installation "
            "may need fixing.\n"
            "Check nvidia-smi before starting GPU training."
        )

    if device == "cpu":
        training_device = "cpu"

    elif cuda_available:
        training_device = 0

        print(
            f"\nGPU detected: {torch.cuda.get_device_name(0)}"
        )

    else:
        training_device = "cpu"

        print("\nWARNING: CUDA is unavailable.")
        print("Training will use the CPU and may be very slow.")

    # --------------------------------------------------------
    # 3. Display training configuration
    # --------------------------------------------------------

    print("\nTRAINING CONFIGURATION")
    print("-" * 40)
    print(f"Epochs       : {EPOCHS}")
    print(f"Patience     : {PATIENCE}")
    print(f"Image size   : {IMAGE_SIZE}")
    print(f"Batch size   : {BATCH_SIZE}")
    print(f"Workers      : {WORKERS}")
    print(f"Device       : {training_device}")
    print(f"Random seed  : {SEED}")
    print("-" * 40)

    # --------------------------------------------------------
    # 4. Load pretrained model
    # --------------------------------------------------------

    print("\nLoading pretrained YOLO model...")

    model = YOLO(MODEL_WEIGHTS)

    # --------------------------------------------------------
    # 5. Start training
    # --------------------------------------------------------

    print("\nStarting training...\n")

    results = model.train(

        # Dataset
        data=str(DATA_YAML),

        # Training duration
        epochs=EPOCHS,
        patience=PATIENCE,

        # Image and batch configuration
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        # Hardware
        device=training_device,
        workers=WORKERS,

        # Output
        project=OUTPUT_DIRECTORY,
        name="buoy_yolo26n_300_epochs",

        # Reproducibility
        seed=SEED,
        deterministic=True,

        # Training options
        pretrained=True,
        save=True,
        save_period=25,
        plots=True,
        val=True,
        verbose=True,

    )

    # --------------------------------------------------------
    # 6. Report training results
    # --------------------------------------------------------

    save_directory = Path(
        getattr(
            results,
            "save_dir",
            Path(OUTPUT_DIRECTORY) / "buoy_yolo26n_300_epochs"
        )
    )

    best_weights = save_directory / "weights" / "best.pt"
    last_weights = save_directory / "weights" / "last.pt"

    print("\n" + "=" * 60)
    print("TRAINING FINISHED")
    print("=" * 60)

    print(f"\nResults directory:\n{save_directory}")

    print(f"\nBest model:\n{best_weights}")

    print(f"\nLast model:\n{last_weights}")

    if best_weights.exists():
        print("\nSUCCESS: Best model weights were saved.")

    else:
        print(
            "\nWARNING: Best weights were not found at the "
            "expected location. Check the training logs."
        )

    print("\nTraining metrics and plots are saved in the run directory.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train a YOLO underwater buoy detector."
    )

    parser.add_argument(
        "--device",
        choices=["auto", "0", "cpu"],
        default="auto",
        help=(
            "auto: use GPU if CUDA is available, otherwise CPU; "
            "0: require NVIDIA GPU; cpu: force CPU."
        ),
    )

    args = parser.parse_args()

    train_model(device=args.device)