"""
Step 2: FastJet Feature Extraction (Vectorized AwkwardClusterSequence API)
===========================================================================
Extract physics observables from jet constituents using FastJet's vectorized API.

Pipeline: Awkward (Vector) -> FastJet (C/A exclusive) -> 6 features

Features:
    1. Jet mass
    2. Jet pT
    3. Jet eta
    4. tau21 = tau2/tau1 (N-subjettiness ratio)
    5. tau32 = tau3/tau2 (N-subjettiness ratio)
    6. D2 (energy correlation ratio)

Algorithm: C/A (cambridge_algorithm) with R=10.0
Clustering: Exclusive (1 jet per event)

Performance: Vectorized AwkwardClusterSequence API processes ALL events
in a single C++ call -- ~1000x faster than per-event Python loop.
"""
import numpy as np
import awkward as ak
import fastjet


def compute_features(events, trim_padding=True):
    """
    Compute 6 FastJet features for each event using vectorized API.

    Parameters
    ----------
    events : ak.Array
        Jagged array of LorentzVectors (from to_awkward.load_awkward)
    trim_padding : bool
        Remove zero-energy constituents before clustering

    Returns
    -------
    features : np.ndarray
        Array of shape (N_events, 6): [mass, pT, eta, tau21, tau32, D2]
    """
    n_events = len(events)

    # Convert to float64 Awkward Array with {px, py, pz, E} fields
    px = ak.values_astype(events.px, np.float64)
    py = ak.values_astype(events.py, np.float64)
    pz = ak.values_astype(events.pz, np.float64)
    E = ak.values_astype(events.E, np.float64)

    cons = ak.zip({
        "px": px,
        "py": py,
        "pz": pz,
        "E": E,
    })

    # C/A jet definition with R=10.0
    jet_def = fastjet.JetDefinition(fastjet.cambridge_algorithm, 10.0)

    # Vectorized clustering -- ALL events in one C++ call
    cs = fastjet._pyjet.AwkwardClusterSequence(cons, jet_def)

    # Get exactly 1 jet per event (exclusive clustering)
    jets = cs.exclusive_jets(n_jets=1)

    # Extract basic kinematics
    mass = ak.to_numpy(jets.mass)
    pt = ak.to_numpy(jets.pt)
    eta = ak.to_numpy(jets.eta)

    # N-subjettiness: returns (N_events, 4) for njets=[1,2,3,4]
    nj = cs.njettiness()
    tau1 = ak.to_numpy(nj[:, 0])
    tau2 = ak.to_numpy(nj[:, 1])
    tau3 = ak.to_numpy(nj[:, 2])

    # Ratios (avoid division by zero)
    tau21 = np.where(tau1 > 1e-10, tau2 / tau1, 1.0)
    tau32 = np.where(tau2 > 1e-10, tau3 / tau2, 1.0)

    # Energy Correlators for D2
    # Use normalized=False to get actual ECF values
    ecf2 = cs.exclusive_jets_energy_correlator(npoint=2, normalized=False)
    ecf3 = cs.exclusive_jets_energy_correlator(npoint=3, normalized=False)

    ecf2_np = ak.to_numpy(ecf2)
    ecf3_np = ak.to_numpy(ecf3)

    # D2 = e3 / (e2)^2
    d2 = np.where(ecf2_np > 1e-10, ecf3_np / (ecf2_np * ecf2_np), 0.0)

    # Stack into (N, 6) array
    features = np.column_stack([mass, pt, eta, tau21, tau32, d2])

    return features.astype(np.float32)


def print_feature_stats(features, label="Features"):
    """Print summary statistics of extracted features."""
    names = ["mass", "pT", "eta", "tau21", "tau32", "D2"]
    print(f"\n  {label}:")
    for i, name in enumerate(names):
        vals = features[:, i]
        print(f"    {name}: mean={np.mean(vals):.4f}, "
              f"std={np.std(vals):.4f}, "
              f"min={np.min(vals):.4f}, "
              f"max={np.max(vals):.4f}")


# ---- Main ----
if __name__ == '__main__':
    print("=" * 60)
    print("PIPELINE STEP 2: FastJet Feature Extraction (Vectorized)")
    print("=" * 60)

    from to_awkward import load_awkward

    for split in ['train.h5', 'val.h5', 'test.h5']:
        print(f"\n--- {split} ---")
        events, labels, weights = load_awkward(split, max_events=100, trim_padding=True)
        feats = compute_features(events)
        print_feature_stats(feats, split.replace('.h5', ''))
        n_top = int(labels.sum())
        if n_top > 0 and n_top < len(labels):
            feats_top = feats[labels == 1]
            feats_qcd = feats[labels == 0]
            print_feature_stats(feats_top, "  Top")
            print_feature_stats(feats_qcd, "  QCD")

    print("\n" + "=" * 60)
    print("  FastJet features ready for DeepSets!")
    print("=" * 60)
