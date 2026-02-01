#!/usr/bin/env python3
"""
Compare ARI results between original and refactored Gauging-delta implementations on flame dataset.
"""

import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import adjusted_rand_score

# Add paths for both implementations
sys.path.insert(0, str(Path(__file__).parent / "legacy"))
sys.path.insert(0, str(Path(__file__).parent / "src"))

def load_flame_data():
    """Load flame dataset."""
    data_path = Path(__file__).parent / "data" / "flame.txt"
    data = np.loadtxt(data_path, delimiter=',')
    X = data[:, :2]  # Features
    y_true = data[:, 2].astype(int) - 1  # Labels (convert to 0-based)
    return X, y_true

def run_original_implementation(X):
    """Run original perception implementation."""
    from perception_original import Perception
    
    perception = Perception()
    labels, _ = perception.fit(X)
    return labels.astype(int)

def run_refactored_implementation(X):
    """Run refactored GaugingDelta implementation."""
    from gauging_delta import GaugingDelta
    
    model = GaugingDelta()
    labels = model.fit(X)
    return labels

def main():
    print("🔥 Comparing ARI on Flame Dataset")
    print("=" * 50)
    
    # Load data
    X, y_true = load_flame_data()
    print(f"Dataset shape: {X.shape}")
    print(f"True clusters: {len(np.unique(y_true))}")
    print(f"True cluster distribution: {np.bincount(y_true)}")
    print()
    
    # Run original implementation
    print("Running original implementation...")
    try:
        original_labels = run_original_implementation(X)
        original_ari = adjusted_rand_score(y_true, original_labels)
        original_n_clusters = len(np.unique(original_labels))
        print(f"✅ Original ARI: {original_ari:.4f}")
        print(f"   Original clusters found: {original_n_clusters}")
        print(f"   Original cluster distribution: {np.bincount(original_labels)}")
    except Exception as e:
        print(f"❌ Original implementation failed: {e}")
        original_ari = None
        original_n_clusters = None
    
    print()
    
    # Run refactored implementation
    print("Running refactored implementation...")
    try:
        refactored_labels = run_refactored_implementation(X)
        refactored_ari = adjusted_rand_score(y_true, refactored_labels)
        refactored_n_clusters = len(np.unique(refactored_labels))
        print(f"✅ Refactored ARI: {refactored_ari:.4f}")
        print(f"   Refactored clusters found: {refactored_n_clusters}")
        print(f"   Refactored cluster distribution: {np.bincount(refactored_labels)}")
    except Exception as e:
        print(f"❌ Refactored implementation failed: {e}")
        refactored_ari = None
        refactored_n_clusters = None
    
    print()
    
    # Compare results
    print("📊 Comparison Results")
    print("-" * 30)
    
    if original_ari is not None and refactored_ari is not None:
        ari_diff = abs(original_ari - refactored_ari)
        print(f"ARI Difference: {ari_diff:.4f}")
        
        if ari_diff < 0.001:
            print("🎉 Perfect parity! ARI scores are essentially identical.")
        elif ari_diff < 0.01:
            print("✅ Very close parity - minor differences only.")
        elif ari_diff < 0.05:
            print("⚠️  Moderate differences - worth investigating.")
        else:
            print("❌ Significant differences - implementation discrepancy.")
        
        print(f"Original ARI: {original_ari:.4f}")
        print(f"Refactored ARI: {refactored_ari:.4f}")
        
        if original_n_clusters != refactored_n_clusters:
            print(f"⚠️  Cluster count difference: {original_n_clusters} vs {refactored_n_clusters}")
    
    elif original_ari is None:
        print("❌ Cannot compare - original implementation failed")
    elif refactored_ari is None:
        print("❌ Cannot compare - refactored implementation failed")

if __name__ == "__main__":
    main()
