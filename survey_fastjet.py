"""
Survey what FastJet Python provides for top tagging features.
"""
import fastjet
from to_awkward import load_awkward

print("=" * 60)
print("FastJet Python API Survey")
print("=" * 60)
print(f"Version: {fastjet.__version__}")
print()

# 1. What built-in taggers are available?
print("--- Built-in Taggers ---")
taggers = ['CASubJetTagger', 'JHTopTagger', 'RestFrameNSubjettinessTagger', 
           'MassDropTagger', 'Filter', 'Pruner', 'Recluster']
for t in taggers:
    obj = getattr(fastjet, t, None)
    if obj:
        print(f"  {t}: AVAILABLE")
    else:
        print(f"  {t}: NOT AVAILABLE")
print()

# 2. What does PseudoJet provide?
print("--- PseudoJet Key Methods ---")
pj = fastjet.PseudoJet(1, 2, 3, 4)
key_methods = ['m', 'm2', 'pt', 'pt2', 'eta', 'rap', 'phi', 'e', 'E',
               'px', 'py', 'pz', 'Et', 'mt', 'delta_R', 'constituents',
               'pieces', 'has_pieces', 'has_constituents']
for m in key_methods:
    if hasattr(pj, m):
        print(f"  {m}: YES")
    else:
        print(f"  {m}: NO")
print()

# 3. Test on real data
print("--- Testing on Real Data (5 events) ---")
events, labels, weights = load_awkward('train.h5', max_events=5, trim_padding=True)

for i in range(5):
    evt = events[i]
    constituents = []
    for p in evt:
        constituents.append(fastjet.PseudoJet(float(p.px), float(p.py), float(p.pz), float(p.E)))
    
    jet_def = fastjet.JetDefinition(fastjet.cambridge_algorithm, 10.0)
    cs = fastjet.ClusterSequence(constituents, jet_def)
    jets = cs.inclusive_jets()
    jet = jets[0]
    
    print(f"\nEvent {i} (label={int(labels[i])}):")
    print(f"  n_const={len(constituents)}, mass={jet.m():.1f}, pt={jet.pt():.1f}, eta={jet.eta():.2f}")
    
    # --- CASubJetTagger ---
    try:
        ca_tagger = fastjet.CASubJetTagger()
        ca_result = ca_tagger(jet)
        print(f"  CASubJetTagger: mass={ca_result.m():.1f}, n_pieces={len(ca_result.pieces())}")
    except Exception as e:
        print(f"  CASubJetTagger FAILED: {e}")
    
    # --- JHTopTagger ---
    try:
        jh_tagger = fastjet.JHTopTagger()
        top_candidate = jh_tagger(jet)
        print(f"  JHTopTagger: mass={top_candidate.m():.1f}, n_pieces={len(top_candidate.pieces())}")
        if len(top_candidate.pieces()) >= 2:
            W = top_candidate.pieces()[0]
            b = top_candidate.pieces()[1]
            print(f"    W mass={W.m():.1f}, b mass={b.m():.1f}")
    except Exception as e:
        print(f"  JHTopTagger FAILED: {e}")
    
    # --- RestFrameNSubjettinessTagger ---
    try:
        rf_def = fastjet.JetDefinition(fastjet.cambridge_algorithm, 0.6)
        rf_tagger = fastjet.RestFrameNSubjettinessTagger(rf_def)
        rf_result = rf_tagger(jet)
        print(f"  RestFrameNSubjettiness: mass={rf_result.m():.1f}, n_pieces={len(rf_result.pieces())}")
    except Exception as e:
        print(f"  RestFrameNSubjettiness FAILED: {e}")
    
    # --- Exclusive subjets (for N-subjettiness axes) ---
    try:
        sub1 = jet.exclusive_subjets_up_to(1)
        sub2 = jet.exclusive_subjets_up_to(2)
        sub3 = jet.exclusive_subjets_up_to(3)
        print(f"  Exclusive subjets: 1={len(sub1)}, 2={len(sub2)}, 3={len(sub3)}")
        if len(sub2) >= 2:
            print(f"    subjet[0]: pt={sub2[0].pt():.1f}, m={sub2[0].m():.1f}")
            print(f"    subjet[1]: pt={sub2[1].pt():.1f}, m={sub2[1].m():.1f}")
    except Exception as e:
        print(f"  Exclusive subjets FAILED: {e}")

print()
print("=" * 60)
print("Survey complete!")
print("=" * 60)
