from ultralytics import YOLO

# Load the best trained model
model = YOLO(
    "/home/krish/Documents/Development/kyushu_project/runs/detect/buoy_yolo26n_300_epochs/weights/best.pt"
)

# Run validation
metrics = model.val(
    data="/home/krish/Documents/kyushu_project/validation.yaml",
    imgsz=640,
    batch=8,
    device=0,
    plots=True
)

# Display results
print("\n===== VALIDATION RESULTS =====")
print(f"mAP50:       {metrics.box.map50:.4f}")
print(f"mAP50-95:    {metrics.box.map:.4f}")
print(f"Precision:   {metrics.box.mp:.4f}")
print(f"Recall:      {metrics.box.mr:.4f}")
