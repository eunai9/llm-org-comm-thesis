#!/usr/bin/env bash
set -euo pipefail
cd ~/projects/thesis
source .venv/bin/activate
set -a
source .env
set +a
mkdir -p logs
log="logs/q1_deepseek_full_current.log"

# Same restart-loop pattern as run_q1_gptoss20b_full.sh, for the same
# reason: NVIDIA's free tier sometimes holds a request without ever
# answering (PROGRESS_nvidia.md section 5), and deepseek-v4.1-flash has
# been observed especially slow to respond (2026-09-25 and 2026-09-29
# sessions both found it timing out where gpt-oss-20b did not). The
# model id itself changed from deepseek-ai/deepseek-v4-flash-0731,
# which NVIDIA retired 2026-09-21 (HTTP 410) -- v4.1-flash is its
# current successor and the two are not cache-compatible.
for i in $(seq 1 30); do
  if python -m thesis.analysis.q1 \
    --nvidia deepseek-ai/deepseek-v4.1-flash \
    --design full \
    --out data/interim/q1_direction_grid_deepseek_full.parquet \
    --progress-every 25 \
    >> "$log" 2>&1
  then
    break
  fi
  echo "$(date '+%Y-%m-%d %H:%M:%S')  restart $i" >> "$log"
  sleep 60
done
