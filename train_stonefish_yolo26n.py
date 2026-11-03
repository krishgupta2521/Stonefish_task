from ultralytics import YOLO
import torch
from pathlib import Path

# =====================================================
# CONFIGURATION
# =====================================================

DATASET_YAML = (
    "/home/krish/Documents/Dataset/stonefish_buoy.v1i.yolo26/data.yaml"
)

MODEL_NAME = "yolo26n.pt"

PROJECT_DIR = (
    "/home/krish/Documents/Development/kyushu_project/"
    "runs/stonefish"
)

RUN_NAME = "buoy_yolo26n_300_epochs"

EPOCHS = 300
PATIENCE = 50
IMAGE_SIZE = 640
BATCH_SIZE = 8

# =====================================================
# GPU CHECK
# =====================================================

print("=" * 55)
print("STONEFISH BUOY DETECTION - YOLO26N TRAINING")
print("=" * 55)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    DEVICE = 0
else:
    print("WARNING: NVIDIA GPU unavailable. Using CPU.")
    DEVICE = "cpu"

# =====================================================
# DATASET CHECK
# =====================================================

yaml_path = Path(DATASET_YAML)

if not yaml_path.is_file():
    raise FileNotFoundError(
        f"Dataset YAML not found: {yaml_path}\n"
        "Update DATASET_YAML to your actual data.yaml path."
    )

# =====================================================
# LOAD MODEL
# =====================================================

print("\nLoading YOLO26n...")

model = YOLO(MODEL_NAME)

# =====================================================
# TRAIN
# =====================================================

print("\nStarting training...")

results = model.train(
    data=str(yaml_path),

    epochs=EPOCHS,
    patience=PATIENCE,

    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,

    project=PROJECT_DIR,
    name=RUN_NAME,

    optimizer="auto",
    pretrained=True,

    workers=4,
    amp=True,

    save=True,
    save_period=25,

    plots=True,
    val=True,

    seed=42,
    deterministic=True,

    verbose=True
)

# =====================================================
# SUMMARY
# =====================================================

best_weights = (
    Path(PROJECT_DIR)
    / RUN_NAME
    / "weights"
    / "best.pt"
)

last_weights = (
    Path(PROJECT_DIR)
    / RUN_NAME
    / "weights"
    / "last.pt"
)

print("\n" + "=" * 55)
print("TRAINING FINISHED")
print("=" * 55)

print(f"Best weights: {best_weights}")
print(f"Last weights: {last_weights}")

if best_weights.exists():
    print("Best model saved successfully.")
else:
    print("Check the training output for errors.")
