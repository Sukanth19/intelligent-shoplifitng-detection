"""
Debug checkpoint to understand NaN issue
"""

import torch
from pathlib import Path

checkpoint_path = Path("checkpoints/fullscale_final.pt")

if not checkpoint_path.exists():
    print(f"❌ Checkpoint not found: {checkpoint_path}")
else:
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("\nCheckpoint contents:")
    for key, value in checkpoint.items():
        if key != 'model_state_dict':
            print(f"  {key}: {value}")
    
    print("\nModel state dict keys:")
    state_dict = checkpoint['model_state_dict']
    for key in list(state_dict.keys())[:10]:
        param = state_dict[key]
        has_nan = torch.isnan(param).any().item()
        has_inf = torch.isinf(param).any().item()
        print(f"  {key}: shape={param.shape}, has_nan={has_nan}, has_inf={has_inf}")
    
    # Check all parameters for NaN/Inf
    total_nan = 0
    total_inf = 0
    for key, param in state_dict.items():
        if torch.isnan(param).any():
            total_nan += 1
            print(f"\n⚠️  NaN found in: {key}")
        if torch.isinf(param).any():
            total_inf += 1
            print(f"\n⚠️  Inf found in: {key}")
    
    print(f"\nSummary:")
    print(f"  Total parameters: {len(state_dict)}")
    print(f"  Parameters with NaN: {total_nan}")
    print(f"  Parameters with Inf: {total_inf}")
    
    if total_nan > 0 or total_inf > 0:
        print("\n❌ Model has NaN/Inf values - training failed!")
        print("This explains why inference produces NaN scores.")
    else:
        print("\n✅ Model parameters look healthy")
