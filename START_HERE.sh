#!/usr/bin/env bash
# Set up the environment (first run only), train the models if missing, start the web app.
set -e
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    .venv/bin/pip install -r requirements.txt
fi

for ds in ucf sht; do
    if [ ! -f "MSL/checkpoints/msl_${ds}_best.pt" ]; then
        echo "Training the ${ds} model (a few minutes on GPU)..."
        (cd MSL && ../.venv/bin/python train.py --dataset "$ds")
    fi
done

echo "Open http://localhost:5000  (Ctrl+C to stop)"
.venv/bin/python app.py
