#!/bin/bash
set -e

export HF_HOME=/root/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1

source /workspace/gemlite_autotune/.venv/bin/activate

echo "[entrypoint] Running prepare script from common_utils."
bash /workspace/common_utils/scripts/prepare.sh
echo "[entrypoint] entrypoint script complete"

# Only exec if arguments are provided
if [ $# -gt 0 ]; then
    echo "[entrypoint] Running exec with arguments: $@"
    exec "$@"
else
    echo "[entrypoint] No arguments provided, script completed successfully"
fi