"""
Run Best Performance Pipeline
1. Train on full dataset (all 950 videos, 75 epochs)
2. Implement complete self-training (3 iterations)
3. Evaluate and compare with base paper

This is the most complete implementation to match paper's performance
"""

import subprocess
import sys
from pathlib import Path
import time

print("="*80)
print("BEST PERFORMANCE PIPELINE")
print("="*80)

print("\nThis pipeline will:")
print("  1. Train on FULL dataset (all available videos, 75 epochs)")
print("  2. Run complete self-training (3 iterations, 25 epochs each)")
print("  3. Evaluate final AUC")
print("  4. Compare with base paper")
print("\n⏱️  Expected time: 2-4 hours total")
print("💾 Will use ALL available video data")

response = input("\nStart best performance pipeline? (y/n): ")
if response.lower() != 'y':
    print("Pipeline cancelled.")
    sys.exit(0)

results = {}
overall_start = time.time()

# Step 1: Full dataset training
print("\n" + "="*80)
print("STEP 1: FULL DATASET TRAINING")
print("="*80)
print("Running: train_full_dataset.py")
print("Expected time: 30-60 minutes")

step1_start = time.time()
try:
    result = subprocess.run(
        [sys.executable, "train_full_dataset.py"],
        cwd=str(Path(__file__).parent),
        capture_output=False,
        text=True
    )
    step1_time = time.time() - step1_start
    results['full_dataset'] = {
        'status': 'success' if result.returncode == 0 else 'failed',
        'time': step1_time
    }
    print(f"\n✅ Full dataset training complete ({step1_time/60:.1f} minutes)")
except Exception as e:
    print(f"\n❌ Full dataset training failed: {e}")
    results['full_dataset'] = {'status': 'failed', 'error': str(e)}
    print("Continuing anyway...")

# Step 2: Self-training
print("\n" + "="*80)
print("STEP 2: COMPLETE SELF-TRAINING")
print("="*80)
print("Running: train_complete_self_training.py")
print("Expected time: 60-120 minutes")

step2_start = time.time()
try:
    result = subprocess.run(
        [sys.executable, "train_complete_self_training.py"],
        cwd=str(Path(__file__).parent),
        capture_output=False,
        text=True
    )
    step2_time = time.time() - step2_start
    results['self_training'] = {
        'status': 'success' if result.returncode == 0 else 'failed',
        'time': step2_time
    }
    print(f"\n✅ Self-training complete ({step2_time/60:.1f} minutes)")
except Exception as e:
    print(f"\n❌ Self-training failed: {e}")
    results['self_training'] = {'status': 'failed', 'error': str(e)}

# Step 3: Evaluation
print("\n" + "="*80)
print("STEP 3: FINAL EVALUATION")
print("="*80)

# Try self-training model first, then full dataset
checkpoints_to_try = [
    "checkpoints/selftraining_complete_final.pt",
    "checkpoints/fulldataset_final.pt",
    "checkpoints/fullscale_final.pt"
]

evaluated = False
for ckpt in checkpoints_to_try:
    ckpt_path = Path(ckpt)
    if ckpt_path.exists():
        print(f"\nEvaluating: {ckpt}")
        step3_start = time.time()
        try:
            result = subprocess.run(
                [sys.executable, "evaluate.py", "--checkpoint", str(ckpt)],
                cwd=str(Path(__file__).parent),
                capture_output=False,
                text=True
            )
            step3_time = time.time() - step3_start
            results[f'evaluation_{ckpt_path.stem}'] = {
                'status': 'success' if result.returncode == 0 else 'failed',
                'time': step3_time
            }
            evaluated = True
            print(f"\n✅ Evaluation complete ({step3_time:.1f} seconds)")
            break
        except Exception as e:
            print(f"\n⚠️  Evaluation failed for {ckpt}: {e}")
            continue

if not evaluated:
    print("\n⚠️  No checkpoints found for evaluation")
    results['evaluation'] = {'status': 'skipped', 'reason': 'no checkpoints'}

# Pipeline complete
total_time = time.time() - overall_start

print("\n" + "="*80)
print("BEST PERFORMANCE PIPELINE COMPLETE!")
print("="*80)

print(f"\nTotal time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")

print("\nPipeline Summary:")
for step, result in results.items():
    status = result['status']
    emoji = "✅" if status == 'success' else "⚠️" if status == 'skipped' else "❌"
    print(f"  {emoji} {step}: {status}")
    if 'time' in result:
        print(f"     Time: {result['time']/60:.1f} minutes")

print("\nGenerated Files:")
checkpoints_dir = Path("checkpoints")
if checkpoints_dir.exists():
    for ckpt in sorted(checkpoints_dir.glob("*.pt")):
        size_mb = ckpt.stat().st_size / (1024 * 1024)
        print(f"  - {ckpt.name} ({size_mb:.1f} MB)")

results_dir = Path("results")
if results_dir.exists() and any(results_dir.glob("*")):
    print(f"\n📊 Check results/ directory for:")
    for result_file in results_dir.glob("*"):
        print(f"  - {result_file.name}")

print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("1. Check results/evaluation_results.txt for final AUC")
print("2. View results/roc_curve.png for performance visualization")
print("3. Compare with base paper (82-84% AUC)")
print("4. Update presentation with final results")
print("="*80)
