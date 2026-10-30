#!/usr/bin/env bash
# Retrain the MSL model(s). Needs the VideoSwin features in data/ (see README).
#
#   ./train.sh                    train both ucf and sht, 40 epochs each
#   ./train.sh ucf                train only ucf
#   ./train.sh ucf 80             train ucf for 80 epochs
#   ./train.sh all 60 --lr 1e-4   any extra flags are passed to MSL/train.py
#
# The current checkpoint is backed up to MSL/checkpoints/old/ before it is overwritten.
set -e
cd "$(dirname "$0")"

dataset="${1:-all}"
epochs="${2:-40}"
shift $(( $# < 2 ? $# : 2 ))

case "$dataset" in
    all) datasets="ucf sht" ;;
    ucf|sht) datasets="$dataset" ;;
    *) echo "usage: $0 [ucf|sht|all] [epochs] [extra train.py flags]"; exit 1 ;;
esac

if [ ! -x .venv/bin/python ]; then
    echo "No .venv found, run ./START_HERE.sh first."
    exit 1
fi

mkdir -p MSL/checkpoints/old
for ds in $datasets; do
    ckpt="MSL/checkpoints/msl_${ds}_best.pt"
    if [ -f "$ckpt" ]; then
        cp "$ckpt" "MSL/checkpoints/old/msl_${ds}_best_$(date +%Y%m%d_%H%M%S).pt"
    fi
    echo "Training ${ds} for ${epochs} epochs..."
    (cd MSL && ../.venv/bin/python train.py --dataset "$ds" --epochs "$epochs" "$@")
done

echo "Done. Results are in MSL/results/. Restart the app to use the new model."
