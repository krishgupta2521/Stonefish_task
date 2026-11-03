#!/usr/bin/env bash
# Start the KYUBIC simulator container with its windows on this screen.
#
#   docker/run.sh [image] [command ...]
#
#   docker/run.sh                                   shell in the prebuilt image
#   docker/run.sh kyubic-sim                        shell in an image you built
#   docker/run.sh kyubic-sim ./start_pipeline.sh --case 1 --no-camera
#
# Default image: ghcr.io/krishgupta2521/stonefish_task:latest
# Passes through, when present: the NVIDIA GPU, Intel/AMD graphics (/dev/dri),
# USB cameras (/dev/video*) and game controllers (/dev/input).
# KYUBIC_SOFTWARE_GL=1  render without a GPU (slow; for machines without one)
# KYUBIC_OUTPUT=DIR     where logs and test results go (default ./kyubic_output)
set -eu

IMAGE="${1:-ghcr.io/krishgupta2521/stonefish_task:latest}"
[ $# -gt 0 ] && shift
OUT="${KYUBIC_OUTPUT:-$PWD/kyubic_output}"
mkdir -p "$OUT/logs" "$OUT/results"

args=(--rm -it --net=host --ipc=host
      -e DISPLAY="${DISPLAY:-:0}"
      -e QT_X11_NO_MITSHM=1
      -v /tmp/.X11-unix:/tmp/.X11-unix:rw
      -v "$OUT/logs:/root/kyushu_project/logs"
      -v "$OUT/results:/root/kyushu_project/camera_rov_project/stonefish/tests/results")

# Let the container open windows on this screen (X11, also under XWayland)
if command -v xhost > /dev/null; then
    xhost +local:root > /dev/null
fi

# Graphics: NVIDIA (needs the NVIDIA Container Toolkit), else Intel/AMD via /dev/dri
if [ "${KYUBIC_SOFTWARE_GL:-0}" = "1" ]; then
    args+=(-e LIBGL_ALWAYS_SOFTWARE=1)
elif command -v nvidia-smi > /dev/null && docker info 2> /dev/null | grep -qi nvidia; then
    args+=(--gpus all)
elif [ -d /dev/dri ]; then
    args+=(--device /dev/dri)
fi

# USB cameras
for v in /dev/video*; do
    [ -e "$v" ] && args+=(--device "$v")
done

# Game controllers: SDL reads /dev/input/event* (character devices, major 13)
if [ -d /dev/input ]; then
    args+=(-v /dev/input:/dev/input --device-cgroup-rule='c 13:* rmw')
fi

exec docker run "${args[@]}" "$IMAGE" "$@"
