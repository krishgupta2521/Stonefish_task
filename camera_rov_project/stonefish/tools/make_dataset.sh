#!/usr/bin/env bash
# Generate a labelled YOLO dataset of the buoy in the simulator.
#
#   tools/make_dataset.sh <out_dir> [count] [prefix] [water_type] [sun_elevation_deg]
#
# Runs the ROV case in dataset mode (TASK2_DATASET=1): the robot turns slowly
# and sweeps its depth, the buoy jumps to random places in view, and
# tools/dataset_saver.py writes images + exact labels. Uses the simulator's
# sensors for the robot's own motion (dataset generation only).
set -u
OUT="${1:?usage: make_dataset.sh <out_dir> [count] [prefix] [water_type] [sun_elevation_deg]}"
COUNT="${2:-1500}"
PREFIX="${3:-sim}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
mkdir -p "$OUT"

set +u
source /opt/ros/jazzy/setup.bash
source "$ROOT/../ros2_ws/install/setup.bash"
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-88}"

python3 "$HERE/dataset_saver.py" "$OUT" --count "$COUNT" --prefix "$PREFIX" > "$OUT/saver_$PREFIX.log" 2>&1 &
SAVER=$!
# The ROV depth hold needs the camera depth estimate to be alive.
python3 "$ROOT/../../underwater_project/visual_heave_estimator.py" > "$OUT/heave_$PREFIX.log" 2>&1 &
FLOW=$!
cd "$ROOT/build"
env TASK2_DATASET=1 TASK2_NAV_SOURCE=sensor TASK2_DEPTH_SOURCE=sensor \
    ${4:+TASK2_WATER_TYPE=$4} ${5:+TASK2_SUN_ELEV=$5} \
    ./Task2Baseline rov > "$OUT/sim_$PREFIX.log" 2>&1 &
SIM=$!
wait "$SAVER"
kill "$SIM" "$FLOW" 2>/dev/null; sleep 2; kill -9 "$SIM" "$FLOW" 2>/dev/null
wait 2>/dev/null
tail -2 "$OUT/saver_$PREFIX.log"
