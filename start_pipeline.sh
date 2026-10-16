#!/usr/bin/env bash
# Start the whole camera -> detection -> simulation chain from kyushu_project/.
#
#   1. external camera publisher   USB camera          -> /external_camera/image_raw
#   2. real camera YOLO detector   /external_camera/image_raw
#                                   -> /object/position (moves the sim target in case 6)
#                                   -> /external_camera/detection_image
#   3. rqt_image_view               real + Stonefish detection images
#   4. Stonefish YOLO detector      /Stonefish/camera/image_raw
#                                   -> /stonefish/target/detection + /stonefish/target/bbox
#                                   -> /stonefish/detection_image
#   5. Stonefish simulation         robot follows /stonefish/target/detection
#
# Usage: ./start_pipeline.sh [--case N] [--device /dev/videoX] [--no-camera] [--no-rqt]
#   --case N      simulation test case 1-7 (default 6: target driven by the real camera)
#   --device D    camera device (default: Innomaker USB camera, else /dev/video0)
#   --no-camera   skip stages 1-2 (cases 1-5 and 7 do not need the real camera)
#   --no-rqt      skip stage 3
# Close the Stonefish window or press Ctrl+C to stop everything.
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
CASE=6
DEVICE=""
USE_CAMERA=1
USE_RQT=1

while [ $# -gt 0 ]; do
    case "$1" in
        --case) CASE="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --no-camera) USE_CAMERA=0; shift ;;
        --no-rqt) USE_RQT=0; shift ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

YOLO_PY="$ROOT/yolo_env/bin/python"
SIM_DIR="$ROOT/camera_rov_project/stonefish/build"
LOG_DIR="$ROOT/logs/pipeline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

set +u
source /opt/ros/jazzy/setup.bash
set -u

for f in "$YOLO_PY" "$SIM_DIR/Task2Baseline"; do
    [ -x "$f" ] || { echo "Missing: $f"; exit 1; }
done

# Only one node may publish /stonefish/target/detection. Match real
# programs only (the simulator binary, or python running a detector script),
# not any command line that merely mentions these names.
DETECTOR_RE='^[^ ]*python[0-9.]* [^ ]*(stonefish_object_detector|stonefish_yolo_detector)'
if pgrep -x Task2Baseline > /dev/null || pgrep -f "$DETECTOR_RE" > /dev/null; then
    echo "A detector or simulator is already running:"
    pgrep -ax Task2Baseline
    pgrep -af "$DETECTOR_RE"
    echo "Stop it first (two detectors on one topic would fight)."
    exit 1
fi

# Each stage runs in its own process group so cleanup also stops the
# children that "ros2 run" spawns.
pgids=()
names=()
start() {
    local name="$1"; shift
    setsid "$@" > "$LOG_DIR/$name.log" 2>&1 &
    pgids+=($!)
    names+=("$name")
    echo "  started $name (log: $LOG_DIR/$name.log)"
}
alive() { kill -0 "$1" 2>/dev/null; }
cleanup() {
    echo
    echo "Stopping all stages..."
    for p in "${pgids[@]}"; do kill -INT -- "-$p" 2>/dev/null; done
    sleep 2
    for p in "${pgids[@]}"; do kill -KILL -- "-$p" 2>/dev/null; done
    echo "Logs: $LOG_DIR"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

echo "Logs: $LOG_DIR"

if [ "$USE_CAMERA" = "1" ]; then
    echo "[1/5] external camera publisher"
    if [ -n "$DEVICE" ]; then
        start camera /usr/bin/python3 "$ROOT/external_camera_publisher.py" --device "$DEVICE"
    else
        start camera /usr/bin/python3 "$ROOT/external_camera_publisher.py"
    fi
    sleep 2
    if ! alive "${pgids[-1]}"; then
        echo "  WARNING: camera did not start:"
        tail -3 "$LOG_DIR/camera.log" | sed 's/^/    /'
        echo "  Continuing; the simulated target will not move."
    fi

    echo "[2/5] real camera YOLO detector"
    start real_yolo "$YOLO_PY" "$ROOT/underwater_project/real_camera_yolo_detector.py"
else
    echo "[1/5] [2/5] real camera skipped (--no-camera)"
fi

if [ "$USE_RQT" = "1" ]; then
    echo "[3/5] rqt_image_view"
    [ "$USE_CAMERA" = "1" ] && \
        start rqt_real ros2 run rqt_image_view rqt_image_view /external_camera/detection_image
    start rqt_stonefish ros2 run rqt_image_view rqt_image_view /stonefish/detection_image
else
    echo "[3/5] rqt skipped (--no-rqt)"
fi

echo "[4/5] Stonefish YOLO detector"
start stonefish_yolo "$YOLO_PY" "$ROOT/underwater_project/stonefish_yolo_detector.py" \
    --mode stonefish \
    --node-name stonefish_yolo_detector \
    --image-topic /Stonefish/camera/image_raw \
    --detection-topic /stonefish/target/detection \
    --bbox-topic /stonefish/target/bbox \
    --output-image-topic /stonefish/detection_image \
    --no-window
STONEFISH_YOLO_PGID="${pgids[-1]}"

echo "[5/5] Stonefish simulation, case $CASE"
cd "$SIM_DIR"
start stonefish ./Task2Baseline "$CASE"
SIM_PGID="${pgids[-1]}"

echo
echo "Running. Close the Stonefish window or press Ctrl+C to stop."
echo "  sim log:  tail -f $LOG_DIR/stonefish.log | grep -v 'AUV ODOM'"

# Stop everything when the simulator closes. If the Stonefish detector dies
# the robot holds position and then surfaces on its own (perception fault).
warned=0
while alive "$SIM_PGID"; do
    if [ "$warned" = "0" ] && ! alive "$STONEFISH_YOLO_PGID"; then
        echo "WARNING: Stonefish YOLO detector stopped; the robot will hold, then surface:"
        tail -5 "$LOG_DIR/stonefish_yolo.log" | sed 's/^/    /'
        warned=1
    fi
    sleep 1
done
