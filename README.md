# Stonefish_task

Autonomous target approach for the KYUBIC V2 AUV in the Stonefish simulator
(ROS 2 Jazzy): YOLO detection of a buoy, gentle approach, stop in front of it, surface.

- `camera_rov_project/stonefish/` - simulator scenario and controller (`src/Task2Manager.cpp`),
  KYUBIC model generated from the CAD (`models/kyubic`, `tools/kyubic_step_export.py`), test harness (`tests/`).
- `camera_rov_project/ros2_ws/` - ROS 2 package with the HSV detector.
- `underwater_project/` - YOLO detectors (`stonefish_yolo_detector.py`, real camera) and the dataset config.
- `runs/detect/training/stonefish_v1/weights/best.pt` - trained YOLO model.
- `start_pipeline.sh` - starts camera publisher, detectors and simulator together.
- `ROV_Autonomy_Presentation.pptx` - 6-slide presentation.

Not included: the KYUBIC STEP CAD (177 MB, over GitHub's file limit), the Stonefish library build, Python venvs.
Build: `source /opt/ros/jazzy/setup.bash`, then CMake in `camera_rov_project/stonefish/`.
Run a test: `DETECTOR=yolo camera_rov_project/stonefish/tests/run_case.sh <case 1-7>`.
