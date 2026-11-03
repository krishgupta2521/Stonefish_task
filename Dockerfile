# KYUBIC camera-only navigation: Stonefish simulator + ROS 2 Jazzy + YOLO detectors.
#
# Build (from the repository root):
#   docker build -t kyubic-sim .                          # PyTorch with CUDA (NVIDIA GPU or CPU)
#   docker build -t kyubic-sim:cpu --build-arg TORCH=cpu .   # CPU-only PyTorch, much smaller
# Run: docker/run.sh kyubic-sim      (see README.md, "Run with Docker")
FROM osrf/ros:jazzy-desktop

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git \
        libglm-dev libsdl2-dev libfreetype-dev libgl-dev \
        python3-venv python3-pip python3-opencv python3-numpy python3-pyqt5 \
        python3-colcon-common-extensions \
        ros-jazzy-cv-bridge ros-jazzy-rqt-image-view \
        mesa-utils v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Stonefish 1.6.0 at the commit this project was developed with. BUILD_TESTS=ON
# gives libStonefish_test.so, the library the simulator links against.
ARG STONEFISH_COMMIT=b21eb8e194c570ff2f61e91aeffb38d73dc25f42
RUN git clone https://github.com/patrykcieslak/stonefish.git /opt/stonefish \
    && git -C /opt/stonefish checkout ${STONEFISH_COMMIT} \
    && cmake -S /opt/stonefish -B /opt/stonefish/build -DBUILD_TESTS=ON \
    && cmake --build /opt/stonefish/build --target Stonefish_test -j"$(nproc)"

# Python for the YOLO detectors. TORCH=cpu installs the CPU-only PyTorch first.
ARG TORCH=cuda
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/yolo_env \
    && if [ "$TORCH" = "cpu" ]; then \
        /opt/yolo_env/bin/pip install --no-cache-dir torch==2.14.1 torchvision==0.29.1 \
            --index-url https://download.pytorch.org/whl/cpu; \
    fi \
    && /opt/yolo_env/bin/pip install --no-cache-dir -r /tmp/requirements.txt
# start_pipeline.sh and tests/run_case.sh use this instead of ./yolo_env
ENV YOLO_PY=/opt/yolo_env/bin/python

# The project: ROS 2 package (HSV detector) and the simulator.
WORKDIR /root/kyushu_project
COPY . .
RUN . /opt/ros/jazzy/setup.sh \
    && cd camera_rov_project/ros2_ws && colcon build \
    && cd ../stonefish \
    && cmake -S . -B build -DSTONEFISH_DIR=/opt/stonefish \
    && cmake --build build -j"$(nproc)"

# OpenGL and CUDA from the host's NVIDIA driver (with the NVIDIA Container Toolkit)
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
    && echo 'source /opt/ros/jazzy/setup.bash' >> /root/.bashrc \
    && echo 'source /root/kyushu_project/camera_rov_project/ros2_ws/install/setup.bash' >> /root/.bashrc
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
