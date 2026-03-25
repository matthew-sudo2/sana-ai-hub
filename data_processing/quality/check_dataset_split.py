import numpy as np
import pickle

# Load the saved feature matrices
good_features = np.load(r'data\synthetic\good_quality_features_real.npy')
bad_features = np.load(r'data\synthetic\bad_quality_features_real.npy')

# Load the metadata
with open(r'data\synthetic\combined_real_data.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"Good quality features shape: {good_features.shape}")
print(f"Bad quality features shape: {bad_features.shape}")
print(f"\nTotal datasets: {good_features.shape[0] + bad_features.shape[0]}")
print(f"Good: {good_features.shape[0]}")
print(f"Bad: {bad_features.shape[0]}")

if 'good_files' in data:
    print("\n✅ Good datasets:")
    for f in data['good_files']:
        print(f"   - {f}")

if 'bad_files' in data:
    print("\n❌ Bad datasets:")
    for f in data['bad_files']:
        print(f"   - {f}")
