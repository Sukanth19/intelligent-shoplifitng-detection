"""
Unit tests for MSL model components
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models import MSLModel, TransformerEncoder, MultiSequenceLearning

print("=" * 80)
print("RUNNING MODEL TESTS")
print("=" * 80)

tests_passed = 0
tests_failed = 0

def test(name, condition, message=""):
    """Simple test function"""
    global tests_passed, tests_failed
    if condition:
        print(f"✓ {name}")
        tests_passed += 1
    else:
        print(f"✗ {name}: {message}")
        tests_failed += 1

# Test 1: Transformer Encoder
print("\n1. Testing Transformer Encoder...")
try:
    encoder = TransformerEncoder(
        d_model=512,
        num_layers=2,
        num_heads=8,
        dim_feedforward=1024,
        dropout=0.1
    )
    x = torch.randn(2, 50, 512)
    output = encoder(x)
    test("Transformer creation", True)
    test("Transformer forward", output.shape == x.shape)
    test("Transformer parameters", sum(p.numel() for p in encoder.parameters()) > 0)
except Exception as e:
    test("Transformer Encoder", False, str(e))

# Test 2: Multi-Sequence Learning
print("\n2. Testing Multi-Sequence Learning...")
try:
    msl = MultiSequenceLearning(sequence_length=32, stride=16)
    features = torch.randn(4, 100, 512)
    sequences, seq_mask, num_seq = msl(features)
    
    test("MSL creation", True)
    test("MSL output shape", sequences.shape[0] == 4 and sequences.shape[2] == 32)
    test("MSL num sequences", num_seq.shape[0] == 4)
    test("MSL mask shape", seq_mask.shape[0:2] == sequences.shape[0:2])
except Exception as e:
    test("Multi-Sequence Learning", False, str(e))

# Test 3: MSL Full Model
print("\n3. Testing MSL Full Model...")
try:
    model = MSLModel(
        input_dim=1024,
        hidden_dim=512,
        num_layers=2,
        num_heads=8,
        sequence_length=32,
        sequence_stride=16
    )
    
    batch_size = 4
    T = 100
    features = torch.randn(batch_size, T, 1024)
    output = model(features)
    
    test("MSL Model creation", True)
    test("MSL Model forward", isinstance(output, dict))
    test("Video logits shape", output['video_logits'].shape == (batch_size, 2))
    test("Video scores shape", output['video_scores'].shape == (batch_size,))
    test("Snippet scores shape", output['snippet_scores'].shape == (batch_size, T))
    test("Sequence features shape", len(output['sequence_features'].shape) == 4)
except Exception as e:
    test("MSL Full Model", False, str(e))

# Test 4: Variable Length Videos
print("\n4. Testing Variable Length Videos...")
try:
    model = MSLModel(input_dim=1024, hidden_dim=256, num_layers=2, num_heads=4)
    
    batch_size = 3
    T = 100
    features = torch.randn(batch_size, T, 1024)
    
    # Create variable length mask
    lengths = torch.tensor([100, 75, 50])
    mask = torch.zeros(batch_size, T, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    
    output = model(features, mask)
    
    test("Variable lengths forward", isinstance(output, dict))
    test("Variable lengths video scores", output['video_scores'].shape == (batch_size,))
    test("Variable lengths num sequences", (output['num_sequences'] > 0).all().item())
except Exception as e:
    test("Variable Length Videos", False, str(e))

# Test 5: Gradient Flow
print("\n5. Testing Gradient Flow...")
try:
    model = MSLModel(input_dim=1024, hidden_dim=256, num_layers=2, num_heads=4)
    features = torch.randn(2, 50, 1024)
    
    output = model(features)
    loss = output['video_logits'].sum() + output['snippet_scores'].sum()
    loss.backward()
    
    # Check if gradients exist
    has_grads = all(
        p.grad is not None
        for p in model.parameters()
        if p.requires_grad
    )
    
    test("Gradient computation", has_grads)
    
    # Check gradient magnitudes
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0
    
    test("Gradient magnitudes", avg_grad_norm > 0)
except Exception as e:
    test("Gradient Flow", False, str(e))

# Test 6: Sequence Parameter Update
print("\n6. Testing Sequence Parameter Update...")
try:
    model = MSLModel(input_dim=1024, hidden_dim=256, num_layers=2, num_heads=4, sequence_length=32)
    
    old_length = model.sequence_length
    model.update_sequence_params(48, 24)
    new_length = model.sequence_length
    
    test("Parameter update", old_length != new_length and new_length == 48)
    
    # Test that model still works
    features = torch.randn(2, 100, 1024)
    output = model(features)
    
    test("Model works after update", isinstance(output, dict))
except Exception as e:
    test("Sequence Parameter Update", False, str(e))

# Test 7: Different Aggregation Methods
print("\n7. Testing Different Aggregation Methods...")
for agg_method in ['mean', 'max', 'attention']:
    try:
        model = MSLModel(
            input_dim=1024,
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            aggregation=agg_method
        )
        features = torch.randn(2, 50, 1024)
        output = model(features)
        
        test(f"Aggregation '{agg_method}'", output['video_scores'].shape == (2,))
    except Exception as e:
        test(f"Aggregation '{agg_method}'", False, str(e))

# Test 8: Inference Mode
print("\n8. Testing Inference Mode...")
try:
    model = MSLModel(input_dim=1024, hidden_dim=256, num_layers=2, num_heads=4)
    model.eval()
    
    with torch.no_grad():
        features = torch.randn(2, 50, 1024)
        output = model(features)
    
    test("Inference mode", isinstance(output, dict))
    test("Video scores range", 0 <= output['video_scores'].min() <= output['video_scores'].max() <= 1)
    test("Snippet scores range", 0 <= output['snippet_scores'].min() <= output['snippet_scores'].max() <= 1)
except Exception as e:
    test("Inference Mode", False, str(e))

# Test 9: Batch Size = 1
print("\n9. Testing Batch Size = 1...")
try:
    model = MSLModel(input_dim=1024, hidden_dim=256, num_layers=2, num_heads=4)
    features = torch.randn(1, 100, 1024)
    output = model(features)
    
    test("Batch size 1", output['video_scores'].shape == (1,))
    test("Batch size 1 snippet", output['snippet_scores'].shape == (1, 100))
except Exception as e:
    test("Batch Size = 1", False, str(e))

# Test 10: Short Videos
print("\n10. Testing Short Videos...")
try:
    model = MSLModel(input_dim=1024, hidden_dim=256, num_layers=2, num_heads=4, sequence_length=32)
    
    # Video shorter than sequence length
    T_short = 20
    features = torch.randn(2, T_short, 1024)
    output = model(features)
    
    test("Short video forward", isinstance(output, dict))
    test("Short video snippet scores", output['snippet_scores'].shape == (2, T_short))
    test("Short video num sequences", (output['num_sequences'] > 0).all().item())
except Exception as e:
    test("Short Videos", False, str(e))

# Test 11: Model Parameters Count
print("\n11. Testing Model Parameters...")
try:
    model = MSLModel(input_dim=1024, hidden_dim=512, num_layers=3, num_heads=8)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    test("Parameters count", total_params > 0)
    test("Trainable parameters", trainable_params == total_params)
    test("Reasonable size", 1_000_000 < total_params < 100_000_000)
    
    print(f"   Total parameters: {total_params:,}")
except Exception as e:
    test("Model Parameters", False, str(e))

# Test 12: Output Value Ranges
print("\n12. Testing Output Value Ranges...")
try:
    model = MSLModel(input_dim=1024, hidden_dim=256, num_layers=2, num_heads=4)
    model.eval()
    
    features = torch.randn(4, 100, 1024)
    with torch.no_grad():
        output = model(features)
    
    video_scores = output['video_scores']
    snippet_scores = output['snippet_scores']
    
    test("Video scores in [0,1]", (video_scores >= 0).all() and (video_scores <= 1).all())
    test("Snippet scores in [0,1]", (snippet_scores >= 0).all() and (snippet_scores <= 1).all())
    test("Video logits unbounded", output['video_logits'].abs().max() > 0)
except Exception as e:
    test("Output Value Ranges", False, str(e))

# Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print(f"✓ Passed: {tests_passed}")
print(f"✗ Failed: {tests_failed}")
print(f"Total: {tests_passed + tests_failed}")

if tests_failed == 0:
    print("\n🎉 ALL MODEL TESTS PASSED!")
    sys.exit(0)
else:
    print(f"\n❌ {tests_failed} test(s) failed")
    sys.exit(1)
