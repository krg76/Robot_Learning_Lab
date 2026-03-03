#!/bin/bash

set -e

LOGFILE="training_$(date +%Y%m%d_%H%M%S).log"

ITERS_LIST=(50 100 200)
SCHEDULES=("linear" "cosine" "cosine_with_restarts")

run=1

for iters in "${ITERS_LIST[@]}"
do
  for schedule in "${SCHEDULES[@]}"
  do
    echo "========== Run $run ==========" | tee -a "$LOGFILE"
    echo "iters=$iters | schedule=$schedule" | tee -a "$LOGFILE"
    echo "Start time: $(date)" | tee -a "$LOGFILE"

    python -m scripts.ddpm \
      --mode train \
      --config config/lift_cube.yaml \
      --iters "$iters" \
      --schedule "$schedule" \
      --savename "$schedule" \
      >> "$LOGFILE" 2>&1

    echo "End time: $(date)" | tee -a "$LOGFILE"

    #git commit -a -m "Auto commit: iters=$iters schedule=$schedule"
    #git push -f

    ((run++))
  done
done

echo "All 9 runs completed." | tee -a "$LOGFILE"