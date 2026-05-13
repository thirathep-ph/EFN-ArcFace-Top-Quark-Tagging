"""
Physics Utilities: Optimized FastJet Feature Extraction (Native SoftDrop)
==========================================================================
Vectorized extraction of 8 advanced substructure features.
Uses Native exclusive_jets_softdrop_grooming from FastJet Python wrapper.

Order: [Mass, mSD, Mult, sqrt(d12), sqrt(d23), Tau32, Tau21, D2]
"""
import numpy as np
import awkward as ak
import fastjet

def compute_features(events):
    """
    Compute 8 optimized features using vectorized FastJet API.
    Corrected to use native field name 'msoftdrop'.
    """
    # 1. Prepare constituents
    px = ak.values_astype(events.px, np.float64)
    py = ak.values_astype(events.py, np.float64)
    pz = ak.values_astype(events.pz, np.float64)
    E = ak.values_astype(events.E, np.float64)
    cons = ak.zip({"px": px, "py": py, "pz": pz, "E": E})

    # 2. Cluster Sequence (C/A R=1.0)
    jet_def = fastjet.JetDefinition(fastjet.cambridge_algorithm, 1.0)
    cs = fastjet.ClusterSequence(cons, jet_def)
    
    # --- Native SoftDrop Grooming ---
    # Field names in native output are suffix-appended (e.g., msoftdrop)
    sd_jets = cs.exclusive_jets_softdrop_grooming(
        njets=1, 
        beta=0.0, 
        symmetry_cut=0.1, 
        R0=1.0
    )
    m_sd = ak.to_numpy(sd_jets.msoftdrop)

    # --- Standard Jets & Scales ---
    jets = cs.exclusive_jets(n_jets=1)
    mass = ak.to_numpy(jets.mass)
    multiplicity = ak.to_numpy(ak.num(events)).astype(np.float32)
    
    # Splitting scales
    d12 = ak.to_numpy(np.sqrt(np.abs(cs.exclusive_dmerge(1))))
    d23 = ak.to_numpy(np.sqrt(np.abs(cs.exclusive_dmerge(2))))

    # --- Shape Features ---
    nj = cs.njettiness()
    tau1, tau2, tau3 = ak.to_numpy(nj[:, 0]), ak.to_numpy(nj[:, 1]), ak.to_numpy(nj[:, 2])
    tau21 = np.where(tau1 > 1e-10, tau2 / tau1, 1.0)
    tau32 = np.where(tau2 > 1e-10, tau3 / tau2, 1.0)

    # Energy Correlator D2
    ecf1 = ak.to_numpy(cs.exclusive_jets_energy_correlator(npoint=1, normalized=False))
    ecf2 = ak.to_numpy(cs.exclusive_jets_energy_correlator(npoint=2, normalized=False))
    ecf3 = ak.to_numpy(cs.exclusive_jets_energy_correlator(npoint=3, normalized=False))
    d2 = np.where(ecf2 > 1e-10, ecf3 * (ecf1**3) / (ecf2**3), 0.0)

    # 3. Final Stack (Perfect Order)
    features = np.column_stack([
        mass, m_sd, multiplicity, d12, d23, tau32, tau21, d2
    ])

    return features.astype(np.float32)

def get_lund_coordinates(events):
    """
    Compute Lund Plane coordinates (ln(1/Delta), ln(kt)) for a batch of events.
    Uses C/A de-clustering.
    """
    import fastjet
    jet_def = fastjet.JetDefinition(fastjet.cambridge_algorithm, 1.0)
    cs = fastjet.ClusterSequence(events, jet_def)
    
    # We take the leading jet for each event
    jets = cs.exclusive_jets(n_jets=1)
    
    all_ln_delta = []
    all_ln_kt = []
    
    # Simplified declustering for the primary branch
    # FastJet Python wrapper allows us to access the parents
    # This is a vectorized approach where possible
    # For now, we use a simplified version for visualization
    for i in range(len(events)):
        # Get history or use specialized Lund Generator if available
        # In this wrapper, we can use the splitting scales as a proxy 
        # or manually walk the tree
        pass # To be fully implemented in the next step
    
    return np.array(all_ln_delta), np.array(all_ln_kt)

def print_feature_stats(features, label="Features"):
    names = ["Mass", "mSD", "Mult", "sqrt(d12)", "sqrt(d23)", "Tau32", "Tau21", "D2"]
    print(f"\n  {label}:")
    for i, name in enumerate(names):
        vals = features[:, i]
        print(f"    {name:10s}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")

if __name__ == '__main__':
    from jet_tagging.data.loader import load_awkward
    events, labels, _ = load_awkward('data/top_tagging/test.h5', max_events=1000)
    feats = compute_features(events)
    print_feature_stats(feats[labels==1], "Top Quark")
    print_feature_stats(feats[labels==0], "QCD Jets")
