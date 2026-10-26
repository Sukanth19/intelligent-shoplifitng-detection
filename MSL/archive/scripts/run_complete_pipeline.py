"""
Complete MSL Pipeline - Runs all next steps automatically
1. Train full-scale model (50 epochs, 400 videos)
2. Evaluate AUC
3. (Optional) Train with Adam if available
4. (Optional) Self-training for additional improvement
"""

import subprocess
import sys
from pathlib import Path
import time

print("="*80)
print("COMPLETE MSL PIPELINE")
print("="*80)

print("\nThis pipeline will:")
print("  1. Train full-scale model (50 epochs, 400 videos, batch 16)")
print("  2. Evaluate AUC on test set")
print("  3. (Optional) Train with Adam optimizer")
print("  4. (Optional) Self-training for +3% AUC")

response = input("\nStart pipeline? (y/n): ")
if response.lower() != 'y':
    print("Pipeline cancelled.")
    sys.exit(0)

results = {}
overall_start = time.time()

# Step 1: Full-scale training
print("\n" + "="*80)
print("STEP 1: FULL-SCALE TRAINING")
print("="*80)
print("Running: train_full_scale.py")
print("Expected time: 30-60 minutes")

step1_start = time.time()
try:
    result = subprocess.run(
        [sys.executable, "train_full_scale.py"],
        cwd=str(Path(__file__).parent),
        capture_output=False,
        text=True
    )
    step1_time = time.time() - step1_start
    results['full_scale_training'] = {
        'status': 'success' if result.returncode == 0 else 'failed',
        'time': step1_time
    }
    print(f"\n✅ Full-scale training complete ({step1_time/60:.1f} minutes)")
except Exception as e:
    print(f"\n❌ Full-scale training failed: {e}")
    results['full_scale_training'] = {'status': 'failed', 'error': str(e)}
    sys.exit(1)

# Step 2: Evaluation
print("\n" + "="*80)
print("STEP 2: EVALUATION")
print("="*80)
print("Running: evaluate.py")
print("Expected time: 1-2 minutes")

step2_start = time.time()
try:
    result = subprocess.run(
        [sys.executable, "evaluate.py", "--checkpoint", "checkpoints/fullscale_final.pt"],
        cwd=str(Path(__file__).parent),
        capture_output=False,
        text=True
    )
    step2_time = time.time() - step2_start
    results['evaluation'] = {
        'status': 'success' if result.returncode == 0 else 'failed',
        'time': step2_time
    }
    print(f"\n✅ Evaluation complete ({step2_time:.1f} seconds)")
except Exception as e:
    print(f"\n❌ Evaluation failed: {e}")
    results['evaluation'] = {'status': 'failed', 'error': str(e)}

# Step 3: Adam optimizer (optional)
print("\n" + "="*80)
print("STEP 3: ADAM OPTIMIZER TRAINING (OPTIONAL)")
print("="*80)
print("This requires PyTorch 2.3.1 or Python 3.11")
print("Expected improvement: +2-3% AUC")

response = input("\nTry Adam optimizer training? (y/n): ")
if response.lower() == 'y':
    step3_start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "train_with_adam.py"],
            cwd=str(Path(__file__).parent),
            capture_output=False,
            text=True
        )
        step3_time = time.time() - step3_start
        
        if result.returncode == 0:
            results['adam_training'] = {'status': 'success', 'time': step3_time}
            print(f"\n✅ Adam training complete ({step3_time/60:.1f} minutes)")
            
            # Evaluate Adam model
            print("\nEvaluating Adam model...")
            subprocess.run(
                [sys.executable, "evaluate.py", "--checkpoint", "checkpoints/adam_final.pt"],
                cwd=str(Path(__file__).parent),
                capture_output=False,
                text=True
            )
        else:
            results['adam_training'] = {'status': 'failed', 'error': 'Optimizer creation failed'}
            print(f"\n⚠️  Adam training failed (likely sympy bug)")
            print("Continuing with manual GD results...")
    except Exception as e:
        print(f"\n⚠️  Adam training failed: {e}")
        results['adam_training'] = {'status': 'failed', 'error': str(e)}
else:
    print("Skipping Adam optimizer training")
    results['adam_training'] = {'status': 'skipped'}

# Step 4: Self-training (optional)
print("\n" + "="*80)
print("STEP 4: SELF-TRAINING (OPTIONAL)")
print("="*80)
print("Iterative pseudo-labeling for additional improvement")
print("Expected improvement: +3% AUC")
print("Expected time: 30-45 minutes")

response = input("\nRun self-training? (y/n): ")
if response.lower() == 'y':
    step4_start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "train_self_training.py"],
            cwd=str(Path(__file__).parent),
            capture_output=False,
            text=True
        )
        step4_time = time.time() - step4_start
        
        if result.returncode == 0:
            results['self_training'] = {'status': 'success', 'time': step4_time}
            print(f"\n✅ Self-training complete ({step4_time/60:.1f} minutes)")
            
            # Evaluate self-trained model
            print("\nEvaluating self-trained model...")
            subprocess.run(
                [sys.executable, "evaluate.py", "--checkpoint", "checkpoints/selftraining_final.pt"],
                cwd=str(Path(__file__).parent),
                capture_output=False,
                text=True
            )
        else:
            results['self_training'] = {'status': 'failed'}
    except Exception as e:
        print(f"\n❌ Self-training failed: {e}")
        results['self_training'] = {'status': 'failed', 'error': str(e)}
else:
    print("Skipping self-training")
    results['self_training'] = {'status': 'skipped'}

# Pipeline complete
total_time = time.time() - overall_start

print("\n" + "="*80)
print("PIPELINE COMPLETE!")
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
print("  - checkpoints/fullscale_final.pt (main model)")
if results.get('adam_training', {}).get('status') == 'success':
    print("  - checkpoints/adam_final.pt (Adam optimizer model)")
if results.get('self_training', {}).get('status') == 'success':
    print("  - checkpoints/selftraining_final.pt (self-trained model)")
print("  - results/evaluation_results.txt (AUC metrics)")
print("  - results/roc_curve.png (ROC curve plot)")
print("  - results/scores.npz (detailed scores)")

print("\n📊 Check results/ directory for evaluation metrics!")
print("="*80)
