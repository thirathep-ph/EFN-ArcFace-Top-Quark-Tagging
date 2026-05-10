import fastjet
import numpy as np

p1 = fastjet.PseudoJet(100.0, 0.0, 0.0, 100.0)
p2 = fastjet.PseudoJet(40.0, 20.0, 30.0, 53.9)
p3 = fastjet.PseudoJet(20.0, -10.0, -5.0, 22.9)

# C/A with R=10 (large enough to capture everything)
ca = fastjet.JetDefinition(fastjet.cambridge_algorithm, 10.0)
cs = fastjet.ClusterSequence([p1, p2, p3], ca)

# Check exclusive_jets signature
import inspect
sig = inspect.signature(cs.exclusive_jets)
print("exclusive_jets signature:", sig)

# Try positional arg
try:
    jets = cs.exclusive_jets(1)
    print(f"exclusive_jets(1): {len(jets)} jets")
    for j, jet in enumerate(jets):
        print(f"  Jet {j}: pt={jet.pt():.1f}, m={jet.m():.1f}")
except Exception as e:
    print(f"exclusive_jets(1) error: {e}")

# Try exclusive_jets_up_to
try:
    jets = cs.exclusive_jets_up_to(3)
    print(f"\nexclusive_jets_up_to(3): {len(jets)} jets")
    for j, jet in enumerate(jets):
        print(f"  Jet {j}: pt={jet.pt():.1f}, m={jet.m():.1f}")
except Exception as e:
    print(f"exclusive_jets_up_to error: {e}")

# Try exclusive_subjets
try:
    # Get the inclusive jet first
    incl = cs.inclusive_jets()[0]
    print(f"\nInclusive jet: pt={incl.pt():.1f}, m={incl.m():.1f}")
    subjets = incl.exclusive_subjets(10.0)
    print(f"exclusive_subjets: {len(subjets)} subjets")
    for j, jet in enumerate(subjets):
        print(f"  Subjet {j}: pt={jet.pt():.1f}, m={jet.m():.1f}")
except Exception as e:
    print(f"exclusive_subjets error: {e}")

# Try exclusive_subjets_up_to
try:
    subjets = incl.exclusive_subjets_up_to(3)
    print(f"\nexclusive_subjets_up_to(3): {len(subjets)} subjets")
    for j, jet in enumerate(subjets):
        print(f"  Subjet {j}: pt={jet.pt():.1f}, m={jet.m():.1f}")
except Exception as e:
    print(f"exclusive_subjets_up_to error: {e}")

# Check dcut for exclusive jets
try:
    # Get the dcut for N exclusive jets
    dcut = cs.exclusive_jets_dcut(2)
    print(f"\nexclusive_jets_dcut(2): {dcut}")
    jets = cs.exclusive_jets(dcut)
    print(f"exclusive_jets(dcut={dcut:.4f}): {len(jets)} jets")
except Exception as e:
    print(f"exclusive_jets_dcut error: {e}")
