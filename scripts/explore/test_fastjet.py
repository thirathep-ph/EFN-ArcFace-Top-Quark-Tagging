"""
Test FastJet chunked processing — small scale.
"""
import os, sys, time
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcjet.data.loader import load_awkward
from arcjet.utils.physics import compute_features

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "top_tagging")
TEST_H5 = os.path.join(DATA_DIR, "test.h5")


def main():
    print("Test FastJet Processing")
    print("=" * 60)
    
    # Test 1: Load 100 events
    print("\nTest 1: 100 events...")
    events, labels, weights = load_awkward(TEST_H5, max_events=100)
    print(f"  Loaded: {len(labels)} events")
    
    try:
        features = compute_features(events)
        print(f"  SUCCESS: {features.shape}")
        print(f"  zg range: [{features[:, 8].min():.4f}, {features[:, 8].max():.4f}]")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Test 2: Load 1000 events
    print("\nTest 2: 1000 events...")
    events, labels, weights = load_awkward(TEST_H5, max_events=1000)
    print(f"  Loaded: {len(labels)} events")
    
    try:
        features = compute_features(events)
        print(f"  SUCCESS: {features.shape}")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Test 3: Load 5000 events
    print("\nTest 3: 5000 events...")
    events, labels, weights = load_awkward(TEST_H5, max_events=5000)
    print(f"  Loaded: {len(labels)} events")
    
    try:
        features = compute_features(events)
        print(f"  SUCCESS: {features.shape}")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Test 4: Load 10000 events
    print("\nTest 4: 10000 events...")
    events, labels, weights = load_awkward(TEST_H5, max_events=10000)
    print(f"  Loaded: {len(labels)} events")
    
    try:
        features = compute_features(events)
        print(f"  SUCCESS: {features.shape}")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Test 5: Load 50000 events
    print("\nTest 5: 50000 events...")
    events, labels, weights = load_awkward(TEST_H5, max_events=50000)
    print(f"  Loaded: {len(labels)} events")
    
    try:
        features = compute_features(events)
        print(f"  SUCCESS: {features.shape}")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Test 6: Load ALL events (404K)
    print("\nTest 6: ALL events (404K)...")
    events, labels, weights = load_awkward(TEST_H5, max_events=999999)
    print(f"  Loaded: {len(labels)} events")
    
    try:
        features = compute_features(events)
        print(f"  SUCCESS: {features.shape}")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Test 7: Chunked processing (1000 events per chunk)
    print("\nTest 7: Chunked processing (1000/chunk, 10000 total)...")
    events, labels, weights = load_awkward(TEST_H5, max_events=10000)
    chunk_size = 1000
    n_chunks = (len(events) + chunk_size - 1) // chunk_size
    all_features = []
    n_failed = 0
    
    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(events))
        chunk = events[start:end]
        
        try:
            feats = compute_features(chunk)
            all_features.append(feats)
        except:
            n_failed += 1
    
    if all_features:
        features = np.vstack(all_features)
        print(f"  SUCCESS: {features.shape}, Failed chunks: {n_failed}")
    else:
        print(f"  ALL FAILED")
    
    print("\n[COMPLETE]")


if __name__ == '__main__':
    main()
