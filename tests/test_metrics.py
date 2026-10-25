import numpy as np
import pytest
from training.logic_hetero import compute_binary_classification_metrics

def test_standard_classification_metrics():
    """Test 1: Verifies correct TP, TN, FP, FN calculation on a standard array."""
    # True classes: [0, 1, 0, 1, 1] (Threshold is 0.5)
    targets = np.array([0.0, 1.0, 0.0, 2.0, 0.5])
    
    # Pred classes: [0, 1, 1, 0, 1]
    preds = np.array([0.1, 0.8, 0.6, 0.4, 0.9])
    
    # TP: 2 (idx 1, 4)
    # TN: 1 (idx 0)
    # FP: 1 (idx 2)
    # FN: 1 (idx 3)
    
    results = compute_binary_classification_metrics(preds, targets, threshold=0.5)
    
    assert results["support_pos"] == 3
    assert results["support_neg"] == 2
    assert np.isclose(results["precision"], 2/3) # TP / (TP + FP)
    assert np.isclose(results["recall"], 2/3)    # TP / (TP + FN)
    assert results["confusion_matrix"][1, 1] == 2 # TP in CM

def test_metrics_nan_filtering():
    """Test 2: Verifies that NaN values in predictions or targets are safely ignored."""
    targets = np.array([0.0, 1.0, np.nan, 2.0])
    preds = np.array([np.nan, 0.8, 0.6, 0.4])
    
    # Only index 1 and 3 are valid for both arrays
    # Valid Targets: [1.0, 2.0] -> Classes [1, 1]
    # Valid Preds:   [0.8, 0.4] -> Classes [1, 0]
    
    results = compute_binary_classification_metrics(preds, targets, threshold=0.5)
    
    assert results["support_pos"] == 2
    assert results["support_neg"] == 0
    assert results["recall"] == 0.5 # 1 TP, 1 FN

def test_metrics_zero_division():
    """Test 3: Verifies that an all-dry dataset doesn't crash the precision/recall math."""
    targets = np.array([0.0, 0.1, 0.2]) # Completely dry (no positive support)
    preds = np.array([0.1, 0.1, 0.1])   # Model accurately predicts all dry
    
    # By default, zero_division=0, so precision and recall should safely return 0.0
    results = compute_binary_classification_metrics(preds, targets, threshold=0.5, zero_division=0)
    
    assert results["support_pos"] == 0
    assert results["support_neg"] == 3
    assert results["precision"] == 0.0
    assert results["recall"] == 0.0
    assert results["confusion_matrix"][0, 0] == 3 # 3 True Negatives