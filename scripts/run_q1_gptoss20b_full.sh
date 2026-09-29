#!/usr/bin/env bash
set -euo pipefail
cd ~/projects/thesis
source .venv/bin/activate
set -a
source .env
set +a
mkdir -p logs
log="logs/q1_gptoss20b_full_current.log"

# NVIDIA's free tier sometimes holds a request without ever answering
# (PROGRESS_nvidia.md section 5). The client already retries each request
# 5 times before giving up; this outer loop catches the case where one
# cell still exhausts all 5 and kills the process. Every finished cell is
# cached, so a restart replays them in seconds and only re-attempts the
# cell that actually failed -- the same pattern that got the deepseek
# pairs run to 183/183 after 3 mid-run stalls (PROGRESS_nvidia.md section 6).
for i in $(seq 1 30); do
  if python -m thesis.analysis.q1 \
    --nvidia openai/gpt-oss-20b@low \
    --design full \
    --out data/interim/q1_direction_grid_gpt_oss_20b_full.parquet \
    --progress-every 25 \
    >> "$log" 2>&1
  then
    break
  fi
  echo "$(date '+%Y-%m-%d %H:%M:%S')  restart $i" >> "$log"
  sleep 60
done
