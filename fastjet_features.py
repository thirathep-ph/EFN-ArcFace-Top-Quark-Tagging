"""
Step 2: FastJet Feature Extraction
====================================
Extract physics observables from jet constituents using FastJet.

Pipeline: Awkward (Vector) -> FastJet (C/A exclusive) -> 6 features

Features:
    1. Jet mass
    2. Jet pT
    3. Jet eta
    4. tau21 = tau2/tau1 (N-subjettiness ratio)
    5. tau32 = tau3/tau2 (N-subjettiness ratio)
    6. D2 (energy correlation ratio)

No R parameter used -- exclusive C/A clustering with large R.
"""
import numpy as np
import awkward as ak
import fastjet


def compute_features(events, trim_padding=True):
    """
    Compute 6 FastJet features for each event.

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
    features = np.zeros((n_events, 6), dtype=np.float32)

    # C/A jet definition with large R (captures everything)
    jet_def = fastjet.JetDefinition(fastjet.cambridge_algorithm, 10.0)

    for i in range(n_events):
        evt = events[i]

        # Convert to list of PseudoJets
        constituents = []
        for p in evt:
            # p has .px, .py, .pz, .E from vector
            constituents.append(fastjet.PseudoJet(float(p.px), float(p.py), float(p.pz), float(p.E)))

        if len(constituents) == 0:
            continue

        # Cluster with C/A
        cs = fastjet.ClusterSequence(constituents, jet_def)

        # Get the hardest inclusive jet
        incl_jets = cs.inclusive_jets()
        if len(incl_jets) == 0:
            continue
        jet = incl_jets[0]

        # Basic jet kinematics
        mass = jet.m()
        pt = jet.pt()
        eta = jet.eta()

        # Get exclusive subjets for N-subjettiness axes
        n_const = len(constituents)
        max_n = min(3, n_const)

        if max_n >= 1:
            subjets_1 = jet.exclusive_subjets_up_to(1)
            subjets_2 = jet.exclusive_subjets_up_to(2) if max_n >= 2 else subjets_1
            subjets_3 = jet.exclusive_subjets_up_to(3) if max_n >= 3 else subjets_2
        else:
            subjets_1 = [jet]
            subjets_2 = [jet]
            subjets_3 = [jet]

        # Compute N-subjettiness
        tau1 = _n_subjettiness(constituents, subjets_1)
        tau2 = _n_subjettiness(constituents, subjets_2) if max_n >= 2 else tau1
        tau3 = _n_subjettiness(constituents, subjets_3) if max_n >= 3 else tau2

        # Ratios (avoid division by zero)
        tau21 = tau2 / tau1 if tau1 > 1e-10 else 1.0
        tau32 = tau3 / tau2 if tau2 > 1e-10 else 1.0

        # Energy correlation functions
        d2 = _d2(constituents)

        features[i] = [mass, pt, eta, tau21, tau32, d2]

    return features


def _n_subjettiness(constituents, axes):
    """
    Compute N-subjettiness tau_N.

    tau_N = (1/d0) * sum_i pT_i * min(dR_i1, dR_i2, ..., dR_iN)

    where d0 = sum_i pT_i * R (normalization), and R is the jet radius.
    We use R = 1.0 as normalization (standard convention).
    """
    if len(constituents) == 0 or len(axes) == 0:
        return 1.0

    # Get axis directions
    ax_eta = np.array([a.eta() for a in axes])
    ax_phi = np.array([a.phi() for a in axes])

    total = 0.0
    d0 = 0.0

    for p in constituents:
        pt_i = p.pt()
        eta_i = p.eta()
        phi_i = p.phi()

        # Min delta-R to any axis
        dphi = np.abs(phi_i - ax_phi)
        dphi = np.minimum(dphi, 2 * np.pi - dphi)
        deta = eta_i - ax_eta
        dr = np.sqrt(deta**2 + dphi**2)
        min_dr = np.min(dr)

        total += pt_i * min_dr
        d0 += pt_i  # normalization: sum pT (R=1 convention)

    if d0 > 1e-10:
        return total / d0
    return 1.0


def _d2(constituents):
    """
    Compute D2 energy correlation function ratio.

    D2 = e3 / (e2)^2

    where:
        e2 = sum_{i<j} pTi pTj * dRij
        e3 = sum_{i<j<k} pTi pTj pTk * dRij * dRik * dRjk

    Uses efficient loop-over-j approach (O(n^3) time but O(n^2) memory).
    """
    n = len(constituents)
    if n < 3:
        return 0.0

    # Extract pT, eta, phi as numpy arrays
    pts = np.array([p.pt() for p in constituents], dtype=np.float64)
    etas = np.array([p.eta() for p in constituents], dtype=np.float64)
    phis = np.array([p.phi() for p in constituents], dtype=np.float64)

    # Compute pairwise delta-R matrix (O(n^2) memory, fine for n~200)
    deta = etas[:, np.newaxis] - etas[np.newaxis, :]
    dphi = phis[:, np.newaxis] - phis[np.newaxis, :]
    dphi = np.abs(dphi)
    dphi = np.minimum(dphi, 2 * np.pi - dphi)
    dr = np.sqrt(deta**2 + dphi**2)

    # e2 = sum_{i<j} pTi * pTj * dRij
    pt_prod_2 = pts[:, np.newaxis] * pts[np.newaxis, :]
    e2 = np.sum(np.triu(pt_prod_2 * dr, k=1))

    # e3 = sum_{i<j<k} pTi * pTj * pTk * dRij * dRik * dRjk
    # Loop over middle index j to avoid O(n^3) memory
    e3 = 0.0
    for j in range(1, n - 1):
        # i < j: indices 0..j-1
        i_slice = slice(0, j)
        # k > j: indices j+1..n-1
        k_slice = slice(j + 1, n)

        # pT_i * pT_j * pT_k for all i<j<k
        # Shape: (j, 1) * (n-j-1,) -> (j, n-j-1)
        pt_ijk = pts[i_slice, np.newaxis] * pts[j] * pts[k_slice]

        # dR_ij * dR_jk * dR_ik for all i<j<k
        dr_ij = dr[i_slice, j]          # (j,)
        dr_jk = dr[j, k_slice]          # (n-j-1,)
        dr_ik = dr[i_slice, k_slice]    # (j, n-j-1)

        # Sum over i and k
        term = pt_ijk * dr_ij[:, np.newaxis] * dr_jk[np.newaxis, :] * dr_ik
        e3 += np.sum(term)

    if e2 > 1e-10:
        return float(e3 / (e2 ** 2))
    return 0.0


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
    print("PIPELINE STEP 2: FastJet Feature Extraction")
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
