@echo off
rem Set up the environment (first run only), train the models if missing, start the web app.
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo Creating virtual environment...
    python -m venv .venv
    .venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    .venv\Scripts\pip install -r requirements.txt
)

if not exist MSL\checkpoints\msl_ucf_best.pt (
    echo Training the UCF-Crime model...
    pushd MSL
    ..\.venv\Scripts\python train.py --dataset ucf
    popd
)
if not exist MSL\checkpoints\msl_sht_best.pt (
    echo Training the ShanghaiTech model...
    pushd MSL
    ..\.venv\Scripts\python train.py --dataset sht
    popd
)

echo Open http://localhost:5000  (Ctrl+C to stop)
.venv\Scripts\python app.py
pause
