import h5py

def main():
    h5_path = 'data/top_tagging/test.h5'
    print(f"Opening HDF5 file: '{h5_path}'...")
    
    with h5py.File(h5_path, 'r') as f:
        print("\nRoot-level keys:")
        for key in f.keys():
            print(f"  - {key}")
            if isinstance(f[key], h5py.Group):
                print(f"    (Group with keys: {list(f[key].keys())})")
            elif isinstance(f[key], h5py.Dataset):
                print(f"    (Dataset with shape: {f[key].shape}, dtype: {f[key].dtype})")
                
        # If there are nested structures, let's explore them
        print("\nDetailed Schema:")
        def print_attrs(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name} | Shape: {obj.shape} | Dtype: {obj.dtype}")
            else:
                print(f"  Group: {name}")
        f.visititems(print_attrs)

if __name__ == '__main__':
    main()
