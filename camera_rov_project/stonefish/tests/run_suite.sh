#!/usr/bin/env bash
# Run every test case with the YOLO detector, two at a time, and summarise.
#
#   tests/run_suite.sh [output_dir]        (default: tests/results/suite_<time>)
#
# Cases 1-7, the no-detector fault (perception_fault) and the missing camera
# depth estimate fault (depth_fault). About 30 minutes.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$HERE/results/suite_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
cd "$HERE/.."

run() { local id=$1; shift; env ROS_DOMAIN_ID="$id" "$@" > /dev/null 2>&1; }

run 41 DETECTOR=yolo tests/run_case.sh 1 "$OUT/case1" 420 & sleep 5
run 42 DETECTOR=yolo tests/run_case.sh 2 "$OUT/case2" 420 & wait
run 43 DETECTOR=yolo tests/run_case.sh 3 "$OUT/case3" 420 & sleep 5
run 44 DETECTOR=yolo tests/run_case.sh 4 "$OUT/case4" 420 & wait
run 45 DETECTOR=yolo tests/run_case.sh 5 "$OUT/case5" 420 & sleep 5
run 46 DETECTOR=yolo tests/run_case.sh 6 "$OUT/case6" 420 & wait
run 47 DETECTOR=yolo tests/run_case.sh 7 "$OUT/case7" 420 & sleep 5
run 60 NO_DETECTOR=1 tests/run_case.sh 1 "$OUT/nodetector" 200 & wait
run 61 DETECTOR=yolo NO_VISION_DEPTH=1 tests/run_case.sh 1 "$OUT/novisiondepth" 200

for r in case1 case2 case3 case4 case5 case6 case7 nodetector novisiondepth; do
    f="$OUT/$r/report.txt"
    echo "== $r: $(tail -1 "$f" 2>/dev/null || echo 'no report')"
    grep -E "FAIL\]|Depth used|true front gap|vertical offset|lateral offset|clearance =|within 1 m" "$f" 2>/dev/null | sed 's/^/   /'
    grep -hE "SAFETY|CALIB\] heave" "$OUT/$r/sim.log" 2>/dev/null | sed -E 's/[0-9]+\.[0-9]+/N/g' | sort | uniq -c | sed 's/^/   /'
done | tee "$OUT/summary.txt"
