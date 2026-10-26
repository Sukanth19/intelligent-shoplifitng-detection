"""
Setup and verification script for MSL project
Run this after providing data paths
"""

import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and report status"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
            return True
        else:
            print(f"✗ {description} failed with return code {result.returncode}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    print("="*80)
    print("MSL PROJECT SETUP")
    print("="*80)
    
    root = Path(__file__).parent
    print(f"\nProject root: {root}")
    
    # Step 1: Check environment
    print("\n" + "="*80)
    print("STEP 1: Environment Check")
    print("="*80)
    run_command("python scripts/check_environment.py", "Environment verification")
    
    # Step 2: Ask about dependencies
    print("\n" + "="*80)
    print("STEP 2: Install Dependencies")
    print("="*80)
    response = input("\nInstall missing dependencies from requirements.txt? (y/n): ")
    if response.lower() == 'y':
        run_command("pip install -r requirements.txt", "Dependency installation")
    else:
        print("Skipped. You can install later with: pip install -r requirements.txt")
    
    # Step 3: Check for data
    print("\n" + "="*80)
    print("STEP 3: Data Verification")
    print("="*80)
    
    data_dir = root / "data"
    ucf_crime = data_dir / "UCF-Crime"
    msl_features = data_dir / "MSL-Features"
    
    print(f"\nChecking data directories:")
    print(f"  UCF-Crime: {ucf_crime}")
    print(f"    Exists: {'✓' if ucf_crime.exists() else '✗'}")
    if ucf_crime.exists():
        try:
            count = len(list(ucf_crime.iterdir()))
            print(f"    Items: {count}")
        except:
            pass
    
    print(f"\n  MSL-Features: {msl_features}")
    print(f"    Exists: {'✓' if msl_features.exists() else '✗'}")
    if msl_features.exists():
        try:
            count = len(list(msl_features.iterdir()))
            print(f"    Items: {count}")
        except:
            pass
    
    # Step 4: Offer to run feature inspection
    if msl_features.exists() and len(list(msl_features.iterdir())) > 0:
        print("\n" + "="*80)
        print("STEP 4: Feature Inspection")
        print("="*80)
        response = input("\nMSL-Features found. Run feature inspection now? (y/n): ")
        if response.lower() == 'y':
            run_command("python scripts/inspect_features.py", "Feature inspection")
        else:
            print("Skipped. You can run later with: python scripts/inspect_features.py")
    else:
        print("\n" + "="*80)
        print("STEP 4: Feature Inspection - SKIPPED")
        print("="*80)
        print("MSL-Features directory not found or empty.")
        print("\nPlease:")
        print("1. Download MSL-VideoSwin-Feature from:")
        print("   https://drive.google.com/drive/folders/12G8HSj7sFK60QHHnGFriFqL4Ld94YgUy")
        print("2. Place in: data/MSL-Features/")
        print("3. Run: python scripts/inspect_features.py")
    
    # Summary
    print("\n" + "="*80)
    print("SETUP SUMMARY")
    print("="*80)
    
    docs_to_read = [
        "README.md - Project overview",
        "PHASE1_COMPLETE.md - Current status",
        "docs/PROGRESS.md - Development tracking",
        "docs/IMPLEMENTATION_NOTES.md - Research fidelity guidelines",
    ]
    
    print("\nKey documents:")
    for doc in docs_to_read:
        print(f"  📄 {doc}")
    
    print("\n" + "="*80)
    print("Setup complete! Check output above for any issues.")
    print("="*80)

if __name__ == "__main__":
    main()
