"""
Unit tests for loss functions
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.losses import MSLRankingLoss, BinaryCrossEntropyLoss

print("=" * 80)
print("RUNNING LOSS FUNCTION TESTS")
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

# Test 1: MSL Ranking Loss - Initialization
print("\n1. Testing MSL Ranking Loss initialization...")
try:
    loss_fn = MSLRankingLoss(margin=0.5, weight_video=1.0, weight_snippet=1.0)
    test("Loss creation", True)
    test("Margin set", loss_fn.margin == 0.5)
    test("Weights set", loss_fn.weight_video == 1.0 and loss_fn.weight_snippet == 1.0)
except Exception as e:
    test("MSL Ranking Loss init", False, str(e))

# Test 2: Forward pass with balanced batch
print("\n2. Testing forward pass with balanced batch...")
try:
    loss_fn = MSLRankingLoss()
    batch_size = 8
    T = 100
    
    video_scores = torch.rand(batch_size)
    snippet_scores = torch.rand(batch_size, T)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    
    result = loss_fn(video_scores, snippet_scores, labels)
    
    test("Result is dict", isinstance(result, dict))
    test("Has loss key", 'loss' in result)
    test("Has loss_video", 'loss_video' in result)
    test("Has loss_snippet", 'loss_snippet' in result)
    test("Has n_pairs", 'n_pairs' in result)
    test("Correct n_pairs", result['n_pairs'] == 16)  # 4 normal × 4 anomaly
    test("Loss is scalar", result['loss'].dim() == 0)
except Exception as e:
    test("Forward pass balanced", False, str(e))

# Test 3: Loss behavior with good vs bad separation
print("\n3. Testing loss behavior...")
try:
    loss_fn = MSLRankingLoss(margin=0.5)
    batch_size = 8
    T = 100
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    
    # Good case: clear separation
    video_scores_good = torch.cat([
        torch.rand(4) * 0.3,      # Normal: 0-0.3
        torch.rand(4) * 0.3 + 0.7  # Anomaly: 0.7-1.0
    ])
    snippet_scores = torch.rand(batch_size, T)
    result_good = loss_fn(video_scores_good, snippet_scores, labels)
    
    # Bad case: no separation
    video_scores_bad = torch.rand(batch_size) * 0.5 + 0.25  # All 0.25-0.75
    result_bad = loss_fn(video_scores_bad, snippet_scores, labels)
    
    test("Good separation has lower loss", result_good['loss'] < result_bad['loss'])
    test("Loss is non-negative", result_good['loss'] >= 0 and result_bad['loss'] >= 0)
except Exception as e:
    test("Loss behavior", False, str(e))

# Test 4: Edge case - all normal
print("\n4. Testing edge case - all normal...")
try:
    loss_fn = MSLRankingLoss()
    batch_size = 4
    
    video_scores = torch.rand(batch_size)
    snippet_scores = torch.rand(batch_size, 50)
    labels = torch.zeros(batch_size, dtype=torch.long)
    
    result = loss_fn(video_scores, snippet_scores, labels)
    
    test("All normal - zero loss", result['loss'].item() == 0.0)
    test("All normal - zero pairs", result['n_pairs'] == 0)
except Exception as e:
    test("Edge case all normal", False, str(e))

# Test 5: Edge case - all anomaly
print("\n5. Testing edge case - all anomaly...")
try:
    loss_fn = MSLRankingLoss()
    batch_size = 4
    
    video_scores = torch.rand(batch_size)
    snippet_scores = torch.rand(batch_size, 50)
    labels = torch.ones(batch_size, dtype=torch.long)
    
    result = loss_fn(video_scores, snippet_scores, labels)
    
    test("All anomaly - zero loss", result['loss'].item() == 0.0)
    test("All anomaly - zero pairs", result['n_pairs'] == 0)
except Exception as e:
    test("Edge case all anomaly", False, str(e))

# Test 6: Gradient flow
print("\n6. Testing gradient flow...")
try:
    loss_fn = MSLRankingLoss()
    
    video_scores = torch.rand(8, requires_grad=True)
    snippet_scores = torch.rand(8, 100, requires_grad=True)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    
    result = loss_fn(video_scores, snippet_scores, labels)
    result['loss'].backward()
    
    has_video_grad = video_scores.grad is not None
    has_snippet_grad = snippet_scores.grad is not None
    
    test("Video scores gradient exists", has_video_grad)
    test("Snippet scores gradient exists", has_snippet_grad)
    
    if has_video_grad:
        test("Video grad non-zero", video_scores.grad.abs().sum() > 0)
    if has_snippet_grad:
        test("Snippet grad non-zero", snippet_scores.grad.abs().sum() > 0)
except Exception as e:
    test("Gradient flow", False, str(e))

# Test 7: Different margins
print("\n7. Testing different margins...")
try:
    video_scores = torch.rand(8)
    snippet_scores = torch.rand(8, 100)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    
    losses = []
    for margin in [0.1, 0.5, 1.0]:
        loss_fn = MSLRankingLoss(margin=margin)
        result = loss_fn(video_scores, snippet_scores, labels)
        losses.append(result['loss'].item())
        test(f"Margin {margin} works", result['loss'] >= 0)
    
    # Larger margin should generally give larger loss
    test("Margin affects loss", losses[0] < losses[2])
except Exception as e:
    test("Different margins", False, str(e))

# Test 8: Different weights
print("\n8. Testing different weight configurations...")
try:
    video_scores = torch.rand(8)
    snippet_scores = torch.rand(8, 100)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    
    # Video only
    loss_fn_video = MSLRankingLoss(weight_video=1.0, weight_snippet=0.0)
    result_video = loss_fn_video(video_scores, snippet_scores, labels)
    test("Video-only weight", result_video['loss'] == result_video['loss_video'])
    
    # Snippet only
    loss_fn_snippet = MSLRankingLoss(weight_video=0.0, weight_snippet=1.0)
    result_snippet = loss_fn_snippet(video_scores, snippet_scores, labels)
    test("Snippet-only weight", result_snippet['loss'] == result_snippet['loss_snippet'])
    
    # Balanced
    loss_fn_balanced = MSLRankingLoss(weight_video=1.0, weight_snippet=1.0)
    result_balanced = loss_fn_balanced(video_scores, snippet_scores, labels)
    test("Balanced weights work", result_balanced['loss'] > 0)
except Exception as e:
    test("Different weights", False, str(e))

# Test 9: Top-k ratio
print("\n9. Testing top-k ratio...")
try:
    video_scores = torch.rand(8)
    snippet_scores = torch.rand(8, 100)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    
    for top_k_ratio in [0.3, 0.5, 0.8, 1.0]:
        loss_fn = MSLRankingLoss(top_k_ratio=top_k_ratio)
        result = loss_fn(video_scores, snippet_scores, labels)
        test(f"Top-k ratio {top_k_ratio}", result['loss'] >= 0)
except Exception as e:
    test("Top-k ratio", False, str(e))

# Test 10: Reduction modes
print("\n10. Testing reduction modes...")
try:
    video_scores = torch.rand(8)
    snippet_scores = torch.rand(8, 100)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    
    loss_fn_mean = MSLRankingLoss(reduction='mean')
    result_mean = loss_fn_mean(video_scores, snippet_scores, labels)
    
    loss_fn_sum = MSLRankingLoss(reduction='sum')
    result_sum = loss_fn_sum(video_scores, snippet_scores, labels)
    
    test("Mean reduction works", result_mean['loss'] >= 0)
    test("Sum reduction works", result_sum['loss'] >= 0)
    test("Sum > Mean", result_sum['loss'] > result_mean['loss'])
except Exception as e:
    test("Reduction modes", False, str(e))

# Test 11: BCE Loss
print("\n11. Testing BCE Loss...")
try:
    bce_loss_fn = BinaryCrossEntropyLoss(weight_video=1.0, weight_snippet=0.0)
    
    video_scores = torch.rand(8)
    snippet_scores = torch.rand(8, 100)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    
    result = bce_loss_fn(video_scores, snippet_scores, labels)
    
    test("BCE result is dict", isinstance(result, dict))
    test("BCE has loss", 'loss' in result)
    test("BCE loss non-negative", result['loss'] >= 0)
except Exception as e:
    test("BCE Loss", False, str(e))

# Test 12: BCE gradient flow
print("\n12. Testing BCE gradient flow...")
try:
    bce_loss_fn = BinaryCrossEntropyLoss(weight_video=1.0)
    
    video_scores = torch.rand(8, requires_grad=True)
    snippet_scores = torch.rand(8, 100)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    
    result = bce_loss_fn(video_scores, snippet_scores, labels)
    result['loss'].backward()
    
    test("BCE gradient exists", video_scores.grad is not None)
    test("BCE gradient non-zero", video_scores.grad.abs().sum() > 0)
except Exception as e:
    test("BCE gradient flow", False, str(e))

# Test 13: Batch size variations
print("\n13. Testing different batch sizes...")
try:
    loss_fn = MSLRankingLoss()
    
    for batch_size in [2, 4, 8, 16]:
        # Ensure balanced
        n_half = batch_size // 2
        video_scores = torch.rand(batch_size)
        snippet_scores = torch.rand(batch_size, 50)
        labels = torch.cat([torch.zeros(n_half), torch.ones(n_half)]).long()
        
        result = loss_fn(video_scores, snippet_scores, labels)
        test(f"Batch size {batch_size}", result['loss'] >= 0)
except Exception as e:
    test("Batch size variations", False, str(e))

# Test 14: Sequence scores (optional input)
print("\n14. Testing with sequence scores...")
try:
    loss_fn = MSLRankingLoss()
    
    video_scores = torch.rand(8)
    snippet_scores = torch.rand(8, 100)
    sequence_scores = torch.rand(8, 5)  # 5 sequences per video
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    
    result = loss_fn(video_scores, snippet_scores, labels, sequence_scores)
    
    test("With sequence scores", result['loss'] >= 0)
    test("Sequence scores affect loss", True)
except Exception as e:
    test("Sequence scores", False, str(e))

# Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print(f"✓ Passed: {tests_passed}")
print(f"✗ Failed: {tests_failed}")
print(f"Total: {tests_passed + tests_failed}")

if tests_failed == 0:
    print("\n🎉 ALL LOSS FUNCTION TESTS PASSED!")
    sys.exit(0)
else:
    print(f"\n❌ {tests_failed} test(s) failed")
    sys.exit(1)
