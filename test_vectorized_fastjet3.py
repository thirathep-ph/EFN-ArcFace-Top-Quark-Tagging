"""
Test the vectorized FastJet API - full feature extraction.
"""
import fastjet
import awkward as ak
import numpy as np
import time
from to_awkward import load_awkward

print("=" * 60)
print("Testing Vectorized FastJet - Full Feature Extraction")
print("=" * 60)

# Load data
events, labels, weights = load_awkward('train.h5', max_events=100, trim_padding=True)
print(f"Loaded {len(events)} events")

# Convert to float64 awkward
px = ak.values_astype(events.x, np.float64)
py = ak.values_astype(events.y, np.float64)
pz = ak.values_astype(events.z, np.float64)
E = ak.values_astype(events.t, np.float64)

cons = ak.zip({
    "px": px,
    "py": py,
    "pz": pz,
    "E": E,
})

# Cluster
jet_def = fastjet.JetDefinition(fastjet.cambridge_algorithm, 10.0)
cs = fastjet._pyjet.AwkwardClusterSequence(cons, jet_def)

# Get inclusive jets
jets = cs.inclusive_jets()
print(f"\nJets: {len(jets)} events, type={ak.type(jets)}")

# Extract mass, pt, eta for each event (first jet)
print("\nFirst 5 jets:")
for i in range(min(5, len(jets))):
    j = jets[i]
    # j is a vector object, access attributes
    mass_val = float(j.mass[0]) if hasattr(j.mass, '__len__') else float(j.mass)
    pt_val = float(j.pt[0]) if hasattr(j.pt, '__len__') else float(j.pt)
    eta_val = float(j.eta[0]) if hasattr(j.eta, '__len__') else float(j.eta)
    print(f"  Event {i}: mass={mass_val:.1f}, pt={pt_val:.1f}, eta={eta_val:.2f}")

# Try njettiness
print("\n--- njettiness() ---")
try:
    nj = cs.njettiness()
    print(f"  njettiness type: {type(nj)}")
    if nj is not None:
        print(f"  njettiness layout: {ak.type(nj)}")
        print(f"  First 3:")
        for i in range(min(3, len(nj))):
            print(f"    Event {i}: {nj[i]}")
except Exception as e:
    print(f"  FAILED: {e}")

# Try energy_correlator
print("\n--- exclusive_jets_energy_correlator() ---")
try:
    ecf = cs.exclusive_jets_energy_correlator()
    print(f"  ECF type: {type(ecf)}")
    if ecf is not None:
        print(f"  ECF layout: {ak.type(ecf)}")
        print(f"  First 3:")
        for i in range(min(3, len(ecf))):
            print(f"    Event {i}: {ecf[i]}")
except Exception as e:
    print(f"  FAILED: {e}")

# Try softdrop
print("\n--- exclusive_jets_softdrop_grooming() ---")
try:
    sd = cs.exclusive_jets_softdrop_grooming()
    print(f"  SD type: {type(sd)}")
    if sd is not None:
        print(f"  SD layout: {ak.type(sd)}")
        print(f"  First 3:")
        for i in range(min(3, len(sd))):
            print(f"    Event {i}: {sd[i]}")
except Exception as e:
    print(f"  FAILED: {e}")

# Try exclusive_subjets_up_to
print("\n--- exclusive_subjets_up_to(3) ---")
try:
    sub = cs.exclusive_subjets_up_to(3)
    print(f"  subjets type: {type(sub)}")
    if sub is not None:
        print(f"  subjets layout: {ak.type(sub)}")
except Exception as e:
    print(f"  FAILED: {e}")

# Speed comparison with larger dataset
print("\n" + "=" * 60)
print("Speed Comparison (1000 events)")
print("=" * 60)

events2, labels2, weights2 = load_awkward('train.h5', max_events=1000, trim_padding=True)
print(f"Loaded {len(events2)} events")

# Convert
px2 = ak.values_astype(events2.x, np.float64)
py2 = ak.values_astype(events2.y, np.float64)
pz2 = ak.values_astype(events2.z, np.float64)
E2 = ak.values_astype(events2.t, np.float64)
cons2 = ak.zip({"px": px2, "py": py2, "pz": pz2, "E": E2})

# Vectorized
t0 = time.time()
cs2 = fastjet._pyjet.AwkwardClusterSequence(cons2, jet_def)
jets2 = cs2.inclusive_jets()
t1 = time.time()
vec_time = t1 - t0
print(f"  Vectorized API: {vec_time:.3f}s ({len(events2)/vec_time:.0f} evts/s)")

# Old per-event (just 100 events for comparison)
events_small, _, _ = load_awkward('train.h5', max_events=100, trim_padding=True)
t0 = time.time()
for i in range(len(events_small)):
    evt = events_small[i]
    cons_list = []
    for p in evt:
        cons_list.append(fastjet.PseudoJet(float(p.px), float(p.py), float(p.pz), float(p.E)))
    cs_old = fastjet.ClusterSequence(cons_list, jet_def)
    jet_old = cs_old.inclusive_jets()[0]
t1 = time.time()
old_time = t1 - t0
print(f"  Old per-event (100 evts): {old_time:.3f}s ({len(events_small)/old_time:.0f} evts/s)")
print(f"  Estimated speedup: {len(events2)/vec_time / (len(events_small)/old_time):.0f}x")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
