"""
Top Tagging Dataset Loader
==========================
Reads H5 files from the Top Tagging dataset and provides
iterator/generator for model training.

Column layout (from H5 metadata):
    values_block_0 has 804 columns:
        [0:800] = 200 constituents x (E, PX, PY, PZ) interleaved
                  particle i: cols 4i+0=E, 4i+1=PX, 4i+2=PY, 4i+3=PZ
        [800:804] = truthE, truthPX, truthPY, truthPZ

    values_block_1 has 2 columns:
        col 0 = ttv (train/test/val indicator: 0=train, 1=test, 2=val)
        col 1 = is_signal_new (actual label: 0=QCD, 1=Top)

Usage:
    from dataset import TopTaggingDataset

    ds = TopTaggingDataset('train.h5')
    for batch_x, batch_y, batch_w in ds.batches(batch_size=256):
        # batch_x: (B, 200, 4) - (E, px, py, pz) for up to 200 constituents
        # batch_y: (B,)         - 0=QCD, 1=Top
        # batch_w: (B,)         - event weights
        ...
"""
import hdf5plugin  # Required: registers HDF5 compression filters
import h5py
import numpy as np

N_CONS = 200


class TopTaggingDataset:
    """
    HDF5 dataset loader for Top Tagging.

    Data layout per entry (804 features):
        [0:800] = 200 constituents x (E, PX, PY, PZ) interleaved
        [800:804] = truthE, truthPX, truthPY, truthPZ
    """

    def __init__(self, path, shuffle=True, seed=42):
        self.path = path
        self.file = h5py.File(path, 'r')
        self.table = self.file['table']['table']
        self.n = self.table.shape[0]

        # Read all labels and weights upfront (small: 2 int64 per entry)
        # values_block_1 has 2 columns:
        #   col 0 = ttv (train/test/val indicator: 0=train, 1=test, 2=val)
        #   col 1 = is_signal_new (actual label: 0=QCD, 1=Top)
        v1 = self.table['values_block_1'][:]
        self._labels = v1[:, 1].astype(np.int8)   # is_signal_new: 0=QCD, 1=Top
        self._weights = np.ones(self.n, dtype=np.float32)  # placeholder, no weights provided

        # Index array
        self._indices = np.arange(self.n)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(self._indices)

        print(f"[dataset] Loaded {path}: {self.n:,} entries")
        n_top = int((self._labels == 1).sum())
        n_qcd = self.n - n_top
        print(f"          Top: {n_top:,} ({n_top/self.n*100:.1f}%), "
              f"QCD: {n_qcd:,} ({n_qcd/self.n*100:.1f}%)")

    def __len__(self):
        return self.n

    @property
    def labels(self):
        return self._labels

    @property
    def weights(self):
        return self._weights

    def get_event(self, idx):
        """Get a single event by index (in shuffled order)."""
        real_idx = self._indices[idx]
        entry = self.table[real_idx]
        return self._parse_entry(entry)

    def get_raw(self, idx):
        """Get a single event by raw table index."""
        entry = self.table[idx]
        return self._parse_entry(entry)

    def _parse_entry(self, entry):
        """Parse a table entry into constituent 4-momenta + truth."""
        v0 = entry['values_block_0']  # (804,)
        # Interleaved (E, px, py, pz) per particle
        es  = v0[0:N_CONS*4:4].copy()   # cols 0,4,8,...,796
        pxs = v0[1:N_CONS*4:4].copy()   # cols 1,5,9,...,797
        pys = v0[2:N_CONS*4:4].copy()   # cols 2,6,10,...,798
        pzs = v0[3:N_CONS*4:4].copy()   # cols 3,7,11,...,799
        truth = v0[800:804].copy()
        constituents = np.stack([es, pxs, pys, pzs], axis=1)  # (200, 4)
        return constituents, truth

    def get_constituents(self, idx_start, idx_end):
        """
        Bulk read constituents for a range of shuffled indices.
        Returns (constituents, labels, weights).
        constituents shape: (N, 200, 4) - (E, px, py, pz)
        """
        real_indices = self._indices[idx_start:idx_end]
        return self.get_constituents_raw(real_indices)

    def get_constituents_raw(self, indices):
        """Bulk read constituents by raw table indices."""
        v0 = self.table['values_block_0'][indices]  # (N, 804)
        # Interleaved extraction: (E, px, py, pz) per particle
        es  = v0[:, 0:N_CONS*4:4]   # (N, 200)
        pxs = v0[:, 1:N_CONS*4:4]   # (N, 200)
        pys = v0[:, 2:N_CONS*4:4]   # (N, 200)
        pzs = v0[:, 3:N_CONS*4:4]   # (N, 200)
        constituents = np.stack([es, pxs, pys, pzs], axis=2)  # (N, 200, 4)
        labels = self._labels[indices]
        weights = self._weights[indices]
        return constituents, labels, weights

    def batches(self, batch_size=256, drop_last=False):
        """
        Generator yielding batches of (x, y, w).

        x: (batch_size, 200, 4) - constituent 4-momenta (E, px, py, pz)
        y: (batch_size,)         - 0=QCD, 1=Top
        w: (batch_size,)         - event weights
        """
        n_batches = self.n // batch_size if drop_last else (self.n + batch_size - 1) // batch_size
        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, self.n)
            if drop_last and end - start < batch_size:
                break
            yield self.get_constituents(start, end)

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def load_all(constituents=True, labels=True, weights=True, max_entries=None):
    """
    Convenience: load all three splits into memory.

    Returns dict with keys: 'train', 'val', 'test'
    Each value is a tuple (constituents, labels, weights).
    If max_entries is set, subsample each split.
    """
    result = {}
    for split in ['train', 'val', 'test']:
        fn = f'{split}.h5'
        ds = TopTaggingDataset(fn, shuffle=False)
        n = min(ds.n, max_entries) if max_entries else ds.n
        result[split] = ds.get_constituents_raw(np.arange(n))
        ds.close()
    return result


# Quick test
if __name__ == '__main__':
    print("Testing dataset loader...")
    ds = TopTaggingDataset('train.h5', shuffle=True)
    print(f"  Total: {len(ds)}")

    x, y, w = ds.get_constituents(0, 5)
    print(f"  Batch x shape: {x.shape}")
    print(f"  Batch y: {y}")
    print(f"  Batch w: {w}")
    print(f"  Sample x[0, :3] (E, px, py, pz):")
    print(f"    E  = {x[0, :3, 0]}")
    print(f"    px = {x[0, :3, 1]}")
    print(f"    py = {x[0, :3, 2]}")
    print(f"    pz = {x[0, :3, 3]}")

    # Verify labels are correct (0=QCD, 1=Top) for all splits
    for split in ['train.h5', 'val.h5', 'test.h5']:
        ds2 = TopTaggingDataset(split, shuffle=False)
        unique, counts = np.unique(ds2._labels, return_counts=True)
        print(f"\n  {split}: labels {dict(zip(unique, counts))}")
        ds2.close()

    ds.close()
    print("\n  Dataset loader ready!")
