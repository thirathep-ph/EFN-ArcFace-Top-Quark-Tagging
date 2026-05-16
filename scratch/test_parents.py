import awkward as ak
import numpy as np
import fastjet
from jet_tagging.data.loader import load_awkward

# 1. Load 1 event
events, _, _ = load_awkward('data/top_tagging/test.h5', max_events=1)
px, py, pz, E = ak.values_astype(events.px, np.float64), ak.values_astype(events.py, np.float64), \
                ak.values_astype(events.pz, np.float64), ak.values_astype(events.E, np.float64)
cons = ak.zip({"px": px, "py": py, "pz": pz, "E": E})

# 2. Cluster
jet_def = fastjet.JetDefinition(fastjet.cambridge_algorithm, 1.5)
cs = fastjet.ClusterSequence(cons, jet_def)
jets = cs.exclusive_jets(n_jets=1)

# 3. Test parents
current_jet = jets[0]
print(f"Has parents: {cs.has_parents(current_jet)}")
if cs.has_parents(current_jet):
    parents = cs.get_parents(current_jet)
    print(f"Parents type: {type(parents)}")
    print(f"Parents fields: {ak.fields(parents)}")
    print(f"Number of parents: {len(parents[0]) if len(parents) > 0 else 0}")
