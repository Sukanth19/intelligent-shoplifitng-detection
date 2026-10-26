# Quick Fix Applied

**Issue**: `train.py` didn't accept `--epochs` argument  
**Error**: `train.py: error: unrecognized arguments: --epochs 50`

## ✅ Fixed

Updated `scripts/train.py` to accept `--epochs` as command-line argument.

### Now you can use:

```bash
# Option 1: Override epochs from command line
python scripts/train.py --config configs/ucf_crime_h5.yaml --epochs 50

# Option 2: Use epochs from config file (no --epochs flag)
python scripts/train.py --config configs/ucf_crime_h5.yaml

# Option 3: Use complete pipeline
python run_complete_pipeline.py --mode full --epochs 50
```

### How it works:

1. If `--epochs` is provided, it overrides the config file
2. If `--epochs` is NOT provided, uses value from config file
3. Config file `training.epochs` is the default

### What was changed:

**File**: `scripts/train.py`

1. Added `--epochs` argument to parser:
```python
parser.add_argument(
    '--epochs',
    type=int,
    default=None,
    help='Number of epochs (overrides config file)'
)
```

2. Override config if provided:
```python
if args.epochs is not None:
    config['training']['epochs'] = args.epochs
```

## ✅ Test the fix:

```bash
# Quick test (2 epochs)
python scripts/train.py --config configs/ucf_crime_h5.yaml --epochs 2

# Full training
python scripts/train.py --config configs/ucf_crime_h5.yaml --epochs 50
```

---

**Status**: ✅ Fixed and ready to use!
