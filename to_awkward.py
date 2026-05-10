"""
Step 1: H5 -> Awkward Array -> Vector
======================================
Convert HDF5 dataset into awkward jagged arrays of
4-vectors for FastJet processing.

Pipeline: H5 Dataset -> NumPy -> Awkward (jagged) -> Vector (Lorentz)

Column layout (from H5 metadata):
    values_block_0 has 804 columns:
        [0:800] = 200 constituents x (E, PX, PY, PZ) interleaved
                  particle i: cols 4i+0=E, 4i+1=PX, 4i+2=PY, 4i+3=PZ
        [800:804] = truthE, truthPX, truthPY, truthPZ

    values_block_1 has 2 columns:
        col 0 = ttv (train/test/val indicator: 0=train, 1=test, 2=val)
        col 1 = is_signal_new (actual label: 0=QCD, 1=Top)
"""
import hdf5plugin
import h5py
import numpy as np
import awkward as ak
import vector

vector.register_awkward()

N_CONS = 200  # max constituents per event


def load_awkward(path, max_events=None, trim_padding=True):
    """
    Load H5 file into awkward arrays of particle 4-vectors.

    Parameters
    ----------
    path : str
        Path to .h5 file
    max_events : int or None
        Limit number of events (for testing)
    trim_padding : bool
        Remove zero-padded constituents (default True)

    Returns
    -------
    events : ak.Array
        Jagged array of LorentzVectors
    labels : np.ndarray
        Binary labels (0=QCD, 1=Top)
    weights : np.ndarray
        Event weights
    """
    f = h5py.File(path, 'r')
    tbl = f['table']['table']
    n = min(tbl.shape[0], max_events) if max_events else tbl.shape[0]

    print(f"  Reading {n:,} entries from {path}...")
    v0 = tbl['values_block_0'][:n]        # (N, 804)
    v1 = tbl['values_block_1'][:n]        # (N, 2)
    f.close()

    # Extract 4-momenta: interleaved (E, px, py, pz) per particle
    # cols: [E_0, PX_0, PY_0, PZ_0, E_1, PX_1, ..., E_199, PX_199, PY_199, PZ_199]
    es  = v0[:, 0:N_CONS*4:4]   # cols 0,4,8,...,796  = E
    pxs = v0[:, 1:N_CONS*4:4]   # cols 1,5,9,...,797  = PX
    pys = v0[:, 2:N_CONS*4:4]   # cols 2,6,10,...,798 = PY
    pzs = v0[:, 3:N_CONS*4:4]   # cols 3,7,11,...,799 = PZ

    # Labels: 0=QCD, 1=Top (from is_signal_new column)
    # values_block_1: col 0 = ttv (0=train, 1=test, 2=val), col 1 = is_signal_new (0=QCD, 1=Top)
    labels = v1[:, 1].astype(np.int8)
    weights = np.ones(n, dtype=np.float32)  # placeholder, no weights provided

    if trim_padding:
        # E=0 means zero-padded (no constituent)
        mask = es > 0
        n_per_event = mask.sum(axis=1)
        total_constituents = int(n_per_event.sum())

        es_flat  = es[mask]
        pxs_flat = pxs[mask]
        pys_flat = pys[mask]
        pzs_flat = pzs[mask]

        counts = n_per_event.tolist()

        e_jagged   = ak.unflatten(es_flat, counts)
        px_jagged  = ak.unflatten(pxs_flat, counts)
        py_jagged  = ak.unflatten(pys_flat, counts)
        pz_jagged  = ak.unflatten(pzs_flat, counts)

        print(f"  Trimmed padding: {total_constituents:,} constituents in {n:,} events "
              f"({total_constituents/n:.1f} avg)")
    else:
        e_jagged  = ak.from_numpy(es)
        px_jagged = ak.from_numpy(pxs)
        py_jagged = ak.from_numpy(pys)
        pz_jagged = ak.from_numpy(pzs)

    # Build Lorentz vectors from (E, px, py, pz)
    events = vector.zip({
        'E': e_jagged,
        'px': px_jagged,
        'py': py_jagged,
        'pz': pz_jagged,
    })

    return events, labels, weights


def print_summary(events, labels):
    """Print a quick summary of the awkward event structure."""
    n_events = len(events)
    n_total = int(ak.sum(ak.num(events)))
    n_per = ak.num(events)

    print(f"  Events: {n_events:,}")
    print(f"  Total constituents: {n_total:,}")
    print(f"  Per event: mean={float(ak.mean(n_per)):.1f}, "
          f"min={int(ak.min(n_per))}, max={int(ak.max(n_per))}")
    print(f"  dtype: {events.type}")

    evt0 = events[0]
    n0 = len(evt0)
    lead = evt0[0]
    print(f"  Event 0: {n0} constituents")
    print(f"  Leading: pT={lead.pt:.1f}, eta={lead.eta:.2f}, phi={lead.phi:.2f}, "
          f"E={lead.E:.1f}, m={lead.mass:.1f}")
    if n0 > 1:
        sub = evt0[1]
        print(f"  Sublead: pT={sub.pt:.1f}, eta={sub.eta:.2f}, phi={sub.phi:.2f}, "
              f"E={sub.E:.1f}, m={sub.mass:.1f}")

    # Check mass near zero (calorimeter particles)
    all_m = ak.flatten(events.mass)
    mean_m = float(ak.mean(all_m))
    max_m = float(ak.max(all_m))
    min_m = float(ak.min(all_m))
    print(f"  Mass: mean={mean_m:.4f}, min={min_m:.4f}, max={max_m:.4f}")

    n_top = int(labels.sum())
    print(f"  Labels: {n_top} Top, {n_events - n_top} QCD")


# ---- Main ----
if __name__ == '__main__':
    print("=" * 60)
    print("PIPELINE STEP 1: H5 -> Awkward -> Vector")
    print("=" * 60)

    for split in ['train.h5', 'val.h5', 'test.h5']:
        print(f"\n--- {split} ---")
        events, labels, weights = load_awkward(split, max_events=10000, trim_padding=True)
        print_summary(events, labels)

    print("\n" + "=" * 60)
    print("  Awkward conversion ready for FastJet!")
    print("=" * 60)
