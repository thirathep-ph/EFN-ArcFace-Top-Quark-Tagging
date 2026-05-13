"""
Step 1: H5 -> Awkward Array -> Vector
======================================
Convert HDF5 dataset into awkward jagged arrays of
4-vectors for FastJet processing and EFN training.

Pipeline: H5 Dataset -> NumPy -> Awkward (jagged) -> Vector (Lorentz)
"""
import hdf5plugin
import h5py
import numpy as np
import awkward as ak
import vector

# Ensure vector is registered with awkward
vector.register_awkward()

N_CONS = 200  # max constituents per event


def load_awkward(path, max_events=None, trim_padding=True):
    """
    Load H5 file into awkward arrays of particle 4-vectors.

    Returns
    -------
    events : ak.Array
        Jagged array of Momentum4D (LorentzVectors)
    labels : np.ndarray
        Binary labels (0=QCD, 1=Top)
    weights : np.ndarray
        Event weights
    """
    f = h5py.File(path, 'r')
    
    # Locate dataset (handle table/table or direct keys)
    if 'table' in f:
        tbl = f['table']['table']
    else:
        tbl = f[list(f.keys())[0]]
        
    n = min(tbl.shape[0], max_events) if max_events else tbl.shape[0]

    print(f"  Reading {n:,} entries from {path}...")
    
    # Layout check: values_block_0 (804 cols) vs values_block_1 (2 cols)
    if 'values_block_0' in tbl.dtype.names:
        v0 = tbl['values_block_0'][:n]
        v1 = tbl['values_block_1'][:n]
        
        # Extract 4-momenta: interleaved (E, px, py, pz) per particle
        es  = v0[:, 0:N_CONS*4:4]
        pxs = v0[:, 1:N_CONS*4:4]
        pys = v0[:, 2:N_CONS*4:4]
        pzs = v0[:, 3:N_CONS*4:4]
        labels = v1[:, 1].astype(np.int8)
    else:
        # Fallback for alternative layouts (e.g. flat array)
        raw = tbl[:n].astype(np.float32)
        es  = raw[:, 0:N_CONS*4:4]
        pxs = raw[:, 1:N_CONS*4:4]
        pys = raw[:, 2:N_CONS*4:4]
        pzs = raw[:, 3:N_CONS*4:4]
        labels = raw[:, 800].astype(np.int8) if raw.shape[1] > 800 else np.zeros(n)

    f.close()
    weights = np.ones(n, dtype=np.float32)

    if trim_padding:
        # E=0 means zero-padded
        mask = es > 0
        n_per_event = mask.sum(axis=1)
        
        es_flat  = es[mask]
        pxs_flat = pxs[mask]
        pys_flat = pys[mask]
        pzs_flat = pzs[mask]

        counts = n_per_event.tolist()
        e_jagged   = ak.unflatten(es_flat, counts)
        px_jagged  = ak.unflatten(pxs_flat, counts)
        py_jagged  = ak.unflatten(pys_flat, counts)
        pz_jagged  = ak.unflatten(pzs_flat, counts)
    else:
        e_jagged  = ak.from_numpy(es)
        px_jagged = ak.from_numpy(pxs)
        py_jagged = ak.from_numpy(pys)
        pz_jagged = ak.from_numpy(pzs)

    # Build Lorentz vectors
    events = ak.zip({
        'E': e_jagged,
        'px': px_jagged,
        'py': py_jagged,
        'pz': pz_jagged,
    }, with_name="Momentum4D")

    return events, labels, weights


def print_summary(events, labels):
    n_events = len(events)
    n_per = ak.num(events)
    print(f"  Events: {n_events:,}")
    print(f"  Constituents: mean={ak.mean(n_per):.1f}, max={ak.max(n_per)}")
    n_top = int(labels.sum())
    print(f"  Labels: {n_top} Top, {n_events - n_top} QCD")


if __name__ == '__main__':
    for split in ['train.h5', 'val.h5', 'test.h5']:
        try:
            events, labels, weights = load_awkward(split, max_events=1000, trim_padding=True)
            print_summary(events, labels)
        except Exception as e:
            print(f"  Could not load {split}: {e}")
