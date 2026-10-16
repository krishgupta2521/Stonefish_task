#!/usr/bin/env bash
# Run one test case end-to-end: simulator + detector (+ fake camera for case 6),
# record telemetry, then analyse it.
#
#   tests/run_case.sh <case 1-7> [output_dir] [max_sim_seconds]
#
# Extra environment:
#   NO_DETECTOR=1   do not start the detector (tests the perception-fault path)
#   DETECTOR=yolo   use underwater_project/stonefish_yolo_detector.py (as
#                   start_pipeline.sh does) instead of the HSV colour detector
#   YOLO_IMGSZ=N    YOLO input size (default: the detector's own)
set -u

CASE="${1:?usage: run_case.sh <case> [output_dir] [max_sim_seconds]}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
OUT="${2:-$HERE/results/case$CASE}"
MAX_TIME="${3:-260}"

mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
rm -f "$OUT"/*.png "$OUT"/*.csv "$OUT"/*.log

set +u
source /opt/ros/jazzy/setup.bash
source "$ROOT/../ros2_ws/install/setup.bash"
set -u
# Own DDS domain per case, so several cases can run side by side.
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-$((40 + CASE))}"

pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null; done; wait 2>/dev/null; }
trap cleanup EXIT

python3 "$HERE/recorder.py" "$OUT" > "$OUT/recorder.log" 2>&1 &
pids+=($!)
KYUSHU="$(cd "$ROOT/../.." && pwd)"
if [ "${NO_DETECTOR:-0}" != "1" ] && [ "${DETECTOR:-hsv}" = "yolo" ]; then
    "$KYUSHU/yolo_env/bin/python" "$KYUSHU/underwater_project/stonefish_yolo_detector.py" \
        --mode stonefish --node-name stonefish_yolo_detector \
        --image-topic /Stonefish/camera/image_raw \
        --detection-topic /stonefish/target/detection \
        --bbox-topic /stonefish/target/bbox \
        --output-image-topic /stonefish/detection_image \
        ${YOLO_IMGSZ:+--imgsz "$YOLO_IMGSZ"} \
        --no-window > "$OUT/detector.log" 2>&1 &
    pids+=($!)
elif [ "${NO_DETECTOR:-0}" != "1" ]; then
    # Executable called directly (not "ros2 run") so cleanup really stops it.
    "$ROOT/../ros2_ws/install/camera_rov/lib/camera_rov/stonefish_object_detector" \
        > "$OUT/detector.log" 2>&1 &
    pids+=($!)
fi
if [ "$CASE" = "6" ]; then
    python3 "$HERE/fake_object_publisher.py" > "$OUT/fake_object.log" 2>&1 &
    pids+=($!)
fi

cd "$ROOT/build"
TASK2_AUTOTEST_MAX_TIME="$MAX_TIME" ./Task2Baseline "$CASE" > "$OUT/sim.log" 2>&1
echo "sim exit code: $?" >> "$OUT/sim.log"
sleep 1
cleanup
trap - EXIT

MODE=""
[ "${NO_DETECTOR:-0}" = "1" ] && MODE="perception_fault"
python3 "$HERE/analyze.py" "$OUT" "$CASE" $MODE | tee "$OUT/report.txt"
