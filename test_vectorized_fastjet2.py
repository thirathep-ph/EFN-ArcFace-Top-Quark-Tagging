"""
Test the vectorized FastJet API - fixed with float64.
"""
import fastjet
import awkward as ak
import numpy as np
import time
from to_awkward import load_awkward

print("=" * 60)
print("Testing Vectorized FastJet API (float64 fix)")
print("=" * 60)

# Load data
events, labels, weights = load_awkward('train.h5', max_events=100, trim_padding=True)
print(f"Loaded {len(events)} events")

# Convert from vector format to plain awkward with float64
print("\nConverting to float64 awkward array...")
t0 = time.time()

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
t1 = time.time()
print(f"  Conversion: {t1-t0:.3f}s")
print(f"  cons type: {ak.type(cons)}")

# Now try AwkwardClusterSequence
print("\n--- Testing AwkwardClusterSequence ---")
jet_def = fastjet.JetDefinition(fastjet.cambridge_algorithm, 10.0)

t0 = time.time()
cs = fastjet._pyjet.AwkwardClusterSequence(cons, jet_def)
t1 = time.time()
print(f"  ClusterSequence creation: {t1-t0:.3f}s")

# Get inclusive jets
t0 = time.time()
jets = cs.inclusive_jets()
t1 = time.time()
print(f"  inclusive_jets(): {t1-t0:.3f}s")
print(f"  jets type: {type(jets)}")
print(f"  jets layout: {ak.type(jets)}")

# Check what we got
print(f"\n  First 5 jets:")
for i in range(min(5, len(jets))):
    j = jets[i]
    print(f"    Event {i}: mass={j.mass:.1f}, pt={j.pt:.1f}, eta={j.eta:.2f}")

# Try njettiness
print("\n--- Testing njettiness() ---")
t0 = time.time()
njettiness = cs.njettiness()
t1 = time.time()
print(f"  njettiness(): {t1-t0:.3f}s")
print(f"  njettiness type: {type(njettiness)}")
if njettiness is not None:
    print(f"  njettiness layout: {ak.type(njettiness)}")
    print(f"  First 5 njettiness values:")
    for i in range(min(5, len(njettiness))):
        print(f"    Event {i}: {njettiness[i]}")

# Try exclusive_jets_energy_correlator
print("\n--- Testing exclusive_jets_energy_correlator() ---")
try:
    t0 = time.time()
    ecf = cs.exclusive_jets_energy_correlator()
    t1 = time.time()
    print(f"  energy_correlator(): {t1-t0:.3f}s")
    print(f"  ecf type: {type(ecf)}")
    if ecf is not None:
        print(f"  ecf layout: {ak.type(ecf)}")
        print(f"  First 5 ECF values:")
        for i in range(min(5, len(ecf))):
            print(f"    Event {i}: {ecf[i]}")
except Exception as e:
    print(f"  FAILED: {e}")

# Try exclusive_jets_softdrop_grooming
print("\n--- Testing exclusive_jets_softdrop_grooming() ---")
try:
    t0 = time.time()
    sd = cs.exclusive_jets_softdrop_grooming()
    t1 = time.time()
    print(f"  softdrop_grooming(): {t1-t0:.3f}s")
    print(f"  sd type: {type(sd)}")
    if sd is not None:
        print(f"  sd layout: {ak.type(sd)}")
except Exception as e:
    print(f"  FAILED: {e}")

# Try exclusive_subjets_up_to
print("\n--- Testing exclusive_subjets_up_to() ---")
try:
    t0 = time.time()
    subjets = cs.exclusive_subjets_up_to(3)
    t1 = time.time()
    print(f"  exclusive_subjets_up_to(3): {t1-t0:.3f}s")
    print(f"  subjets type: {type(subjets)}")
    if subjets is not None:
        print(f"  subjets layout: {ak.type(subjets)}")
except Exception as e:
    print(f"  FAILED: {e}")

# Compare speed with old per-event loop
print("\n--- Speed Comparison (100 events) ---")
# Old method
t0 = time.time()
for i in range(len(events)):
    evt = events[i]
    cons_list = []
    for p in evt:
        cons_list.append(fastjet.PseudoJet(float(p.px), float(p.py), float(p.pz), float(p.E)))
    cs_old = fastjet.ClusterSequence(cons_list, jet_def)
    jet_old = cs_old.inclusive_jets()[0]
    _ = jet_old.m(), jet_old.pt(), jet_old.eta()
t1 = time.time()
old_time = t1 - t0
print(f"  Old per-event loop: {old_time:.3f}s ({len(events)/old_time:.0f} evts/s)")

# New vectorized method
t0 = time.time()
cs_new = fastjet._pyjet.AwkwardClusterSequence(cons, jet_def)
jets_new = cs_new.inclusive_jets()
t1 = time.time()
new_time = t1 - t0
print(f"  New vectorized API: {new_time:.3f}s ({len(events)/new_time:.0f} evts/s)")
print(f"  Speedup: {old_time/new_time:.1f}x")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
