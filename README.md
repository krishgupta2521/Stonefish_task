# KYUBIC camera-only navigation (Stonefish + ROS 2)

A real-size simulation of the KYUBIC V2 AUV that finds, approaches and stops at a buoy using **only its forward camera**: no IMU, DVL or depth sensor is used for control. The robot predicts its motion from the thrust it commands and corrects that prediction with what the camera sees (YOLO buoy detection and optical flow).

- **AUV mode:** calibrate, search, align, approach, stop about 8 cm short, back off, surface.
- **ROV mode:** a pilot drives with a game controller or keyboard; the robot holds depth and heading and will not drive into the buoy.
- **Real camera in the loop:** a yellow object moved in front of a real USB camera moves the simulated buoy.

Simulation test suite: 9 of 9 scenarios pass, including the buoy behind the robot, no buoy at all, and detector failure.

| Folder / file | What it is |
| --- | --- |
| `camera_rov_project/stonefish/` | Simulator scenario and controller (`src/Task2Manager.cpp`), KYUBIC and buoy models (`models/`), test harness (`tests/`), dataset tools (`tools/`) |
| `camera_rov_project/ros2_ws/` | ROS 2 package with a simple colour (HSV) detector |
| `underwater_project/` | YOLO detectors (simulator camera, real camera) and the optical-flow depth estimator |
| `runs/.../weights/best.pt` | Trained YOLO models the detectors load |
| `start_pipeline.sh` | Starts camera, detectors, simulator and logger together |
| `Dockerfile`, `docker/` | Container image with everything installed |
| `KYUBIC_camera_only_navigation.pptx` | Project presentation |

There are two ways to run it: **Docker** (easiest, nothing to install except Docker) or a **native install** on Ubuntu 24.04.

## What you need

- **Linux with a graphics card that supports OpenGL 4.3** (the simulator needs it). An NVIDIA GPU is recommended; Intel or AMD graphics also work.
- Tested on Ubuntu 24.04 with an NVIDIA RTX 3050 laptop GPU.
- Optional: a USB camera (to move the buoy with a real object) and a game controller (ROV mode).

## Option 1: run with Docker

### 1. Install Docker (once)

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER      # then log out and back in
```

**NVIDIA GPU only:** also install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), then:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 2. Get the image

Download the prebuilt image (built automatically from this repository):

```bash
docker pull ghcr.io/krishgupta2521/stonefish_task:latest   # with GPU support (about 10 GB)
docker pull ghcr.io/krishgupta2521/stonefish_task:cpu      # no NVIDIA GPU: smaller
```

Or build it yourself from a clone of this repository (20 to 40 minutes):

```bash
git clone https://github.com/krishgupta2521/Stonefish_task.git
cd Stonefish_task
docker build -t kyubic-sim .                              # or: --build-arg TORCH=cpu
```

### 3. Run

From the repository folder (`docker/run.sh` sets up the window, GPU, camera and controller for you):

```bash
docker/run.sh                                  # opens a shell in the prebuilt image
# or: docker/run.sh kyubic-sim                 # an image you built yourself
```

Inside the container, the same commands as in [Run the simulation](#run-the-simulation) work, for example:

```bash
./start_pipeline.sh --case 1 --no-camera
```

Logs and test results are saved on your machine in `./kyubic_output/`. They are owned by root; `sudo chown -R $USER kyubic_output` fixes that.

**No GPU at all?** `KYUBIC_SOFTWARE_GL=1 docker/run.sh ghcr.io/krishgupta2521/stonefish_task:cpu` renders on the CPU. It runs, but slowly, so the robot's behaviour will not match the results above.

**Windows:** use WSL2 with Ubuntu 24.04 and Docker inside WSL2. Windows' WSLg shows the windows. USB cameras need [usbipd-win](https://learn.microsoft.com/windows/wsl/connect-usb). This setup has not been tested; if the simulator window fails, try `KYUBIC_SOFTWARE_GL=1`.

**macOS:** not supported. Docker on a Mac has no OpenGL 4.3 graphics, and Apple Silicon would have to emulate the image.

## Option 2: native install (Ubuntu 24.04)

### 1. ROS 2 Jazzy

Follow the official guide, [Install ROS 2 Jazzy (deb packages)](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html), and install `ros-jazzy-desktop`. Then:

```bash
echo 'source /opt/ros/jazzy/setup.bash' >> ~/.bashrc
source ~/.bashrc
```

### 2. System packages

```bash
sudo apt install build-essential cmake git libglm-dev libsdl2-dev libfreetype-dev libgl-dev \
    python3-venv python3-opencv python3-numpy python3-pyqt5 python3-colcon-common-extensions \
    ros-jazzy-cv-bridge ros-jazzy-rqt-image-view mesa-utils
glxinfo -B | grep "OpenGL version"        # must be 4.3 or higher
```

### 3. Stonefish simulator library

The project was developed with Stonefish 1.6.0 at commit `b21eb8e`:

```bash
git clone https://github.com/patrykcieslak/stonefish.git ~/stonefish
cd ~/stonefish
git checkout b21eb8e194c570ff2f61e91aeffb38d73dc25f42
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build --target Stonefish_test -j"$(nproc)"
```

`BUILD_TESTS=ON` builds `libStonefish_test.so`, the library this project links against. Keep the `~/stonefish` folder: the simulator loads its shaders from there at run time.

### 4. This repository and its Python environment

```bash
git clone https://github.com/krishgupta2521/Stonefish_task.git ~/Stonefish_task
cd ~/Stonefish_task
python3 -m venv yolo_env
yolo_env/bin/pip install -r requirements.txt
```

The environment must be called `yolo_env` and sit in the repository folder, because the scripts look for it there. To use another one, set `YOLO_PY=/path/to/python`.

### 5. Build

```bash
source /opt/ros/jazzy/setup.bash
cd ~/Stonefish_task/camera_rov_project/ros2_ws
colcon build
cd ../stonefish
cmake -S . -B build -DSTONEFISH_DIR=$HOME/stonefish
cmake --build build -j"$(nproc)"
```

This produces `camera_rov_project/stonefish/build/Task2Baseline`, the simulator.

## Run the simulation

Run everything from the repository folder. Close the Stonefish window or press Ctrl+C to stop.

### Autonomous (AUV) mode

```bash
./start_pipeline.sh --case 1 --no-camera
```

`--no-camera` skips the real USB camera; only case 6 needs it.

| Case | Scenario |
| --- | --- |
| 1 | Buoy straight ahead |
| 2 | Buoy slightly higher |
| 3 | Buoy deeper |
| 4 | Buoy behind the robot (it must search first) |
| 5 | Buoy behind and deeper |
| 6 | Buoy moved by a real object in front of the USB camera (the default) |
| 7 | No buoy: the robot searches 1.4 turns, then surfaces |

### With the real USB camera (case 6)

```bash
./start_pipeline.sh                          # uses the Innomaker USB camera, else /dev/video0
./start_pipeline.sh --device /dev/video2     # another camera
./start_pipeline.sh --video recording.mp4    # a recorded video instead of a camera
```

Move a yellow object in front of the camera; the simulated buoy follows it. Two image windows show what each detector sees.

### ROV (piloted) mode

```bash
./start_pipeline.sh --rov
```

| Action | Game controller | Keyboard (Stonefish window focused) |
| --- | --- | --- |
| Forward / back | left stick up / down | W / S |
| Turn | left stick left / right | A / D |
| Sideways | right stick left / right | Q / E |
| Up / down | right stick up / down | R / F |
| Stop all thrusters | B | Space |

In the autonomous cases, M switches between autonomous and manual driving.

`./start_pipeline.sh --help` lists all options. Each run writes its logs and per-frame latency CSVs to `logs/pipeline_<date>_<time>/`.

## Run the tests

Each test starts the simulator, runs one scenario, and grades it against the simulator's ground truth:

```bash
cd camera_rov_project/stonefish
DETECTOR=yolo tests/run_case.sh 1        # one case; report in tests/results/case1/report.txt
tests/run_suite.sh                       # all 9 runs (cases 1-7 + 2 fault tests), 30-40 minutes
```

Run at most two simulators at the same time; more overloads the camera pipeline. `tests/run_case.sh` explains its other options (for example `NO_DETECTOR=1` to test a detector failure).

## Datasets and training

The datasets are not in the repository because of their size. Download them from the [datasets release](https://github.com/krishgupta2521/Stonefish_task/releases/tag/datasets-v1):

- `sim_buoy_dataset.zip`: 2,656 simulator images with exact labels, for the simulator detector.
- `real_camera_datasets.zip`: the real-camera buoy images.

Unzip them into the repository folder. Then:

```bash
# Make a new simulator dataset (images + exact labels), e.g. 500 frames:
camera_rov_project/stonefish/tools/make_dataset.sh sim_buoy_dataset 500 extra
# Train the simulator detector as it was trained (fine-tuned from the earlier model):
yolo_env/bin/yolo detect train data=sim_buoy_dataset/data.yaml \
    model=runs/stonefish/buoy_yolo26n_300_epochs/weights/best.pt \
    imgsz=960 epochs=60 batch=8 patience=15
```

Set `path:` in each `data.yaml` to where you unzipped it. The training scripts in the repository root (`train_*.py`) point at folders on the author's machine; edit their paths first.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `ROS 2 is not sourced` when building | `source /opt/ros/jazzy/setup.bash`, then build again |
| Simulator window fails, or an OpenGL error | `glxinfo -B` must report OpenGL 4.3 or higher; update the graphics driver |
| Docker: no window appears | Run `xhost +local:root` on the host, and start the container with `docker/run.sh` |
| Docker: NVIDIA GPU not used | Install the NVIDIA Container Toolkit (step 1); `docker info` must list an `nvidia` runtime |
| `numpy` / `cv_bridge` errors | Keep `numpy==1.26.4` (ROS 2 Jazzy's cv_bridge needs numpy 1.x) |
| Detector is slow | Without an NVIDIA GPU, YOLO runs on the CPU; use fewer frames or a GPU |
| Game controller not found | Plug it in before starting; in Docker, start with `docker/run.sh` |

## Credits

- [Stonefish](https://github.com/patrykcieslak/stonefish) simulator by Patryk Cieślak (GPL-3.0).
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0).
- KYUBIC V2 model generated from the team's CAD (`tools/kyubic_step_export.py`). The STEP file itself is not included: at 177 MB it is over GitHub's file size limit.
