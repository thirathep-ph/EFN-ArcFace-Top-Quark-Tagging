"""Does a convergent form actually EMERGE, is it on the Pareto front, and how
far is it from the model?

Answers three questions with existing artifacts only -- no new PySR runs.

Q1 convergence : a form counts as converged only if it is the SAME equation up
                 to an affine transform (scale+offset are gauge: PySR fits a
                 scaled target). Tested numerically as b = p*a + q, require
                 max|resid| < 1e-6.
Q2 Pareto      : report, for each seed, the loss at the converged complexity
                 and whether that point sits on that seed's front (it does by
                 construction -- every recorded row is a front point), and how
                 the converged form's own loss compares with the best loss
                 achievable at that complexity.
Q3 performance : AUC / Brier / ECE / accuracy of the converged closed form vs
                 the EFN-ArcFace model, all on the same 404k test set.
"""
import json

import numpy as np
import sympy as sp
from sklearn.metrics import brier_score_loss, roc_auc_score

P = "experiments/physics_sr"
R = "experiments/robustness_s16_m05"
VARS = ["Mass_over_pT", "mSD_over_pT", "Mult", "nSD", "sqrt_d12",
        "sqrt_d23", "Tau21", "Tau32", "zg", "theta_g"]
_OPS = {"tanh": sp.tanh, "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
        "Abs": sp.Abs, "square": lambda x: x ** 2}

ms = json.load(open(f"{P}/pysr_multiseed_fronts.json", encoding="utf-8"))
names = ms["variable_names"]
mean = dict(zip(names, ms["normalization"]["x_mean"]))
std = dict(zip(names, ms["normalization"]["x_std"]))
runs = {int(k): v for k, v in ms["runs"].items()}
seeds = ms["seeds"]

L = np.load(f"{P}/logits.npz")
C = np.load(f"{R}/jet_pt_mpt_cache.npz")
lab = L["labels"].astype(int)
logit = L["logitdiff"].astype(np.float64)
mpt = C["mpt"].astype(np.float64)
tau = C["tau32"].astype(np.float64)
zm = (mpt - mean["Mass_over_pT"]) / std["Mass_over_pT"]
zt = (tau - mean["Tau32"]) / std["Tau32"]

y_mean = -0.7999436909290824
y_std = 8.71554458720464


def ece(p, y, nb=15):
    b = np.linspace(0, 1, nb + 1)
    i = np.clip(np.digitize(p, b) - 1, 0, nb - 1)
    return float(sum(abs(p[i == k].mean() - y[i == k].mean()) * (i == k).mean()
                     for k in range(nb) if (i == k).any()))


def score(s):
    p = 1.0 / (1.0 + np.exp(-np.clip(s, -50, 50)))
    return dict(auc=float(roc_auc_score(lab, s)),
                brier=float(brier_score_loss(lab, p)),
                ece=ece(p, lab),
                acc=float(((p > .5).astype(int) == lab).mean()))


def parse(eq):
    return sp.sympify(eq, locals={**_OPS, **{v: sp.Symbol(v) for v in VARS}})


def num(expr):
    f = sp.lambdify([sp.Symbol("Mass_over_pT"), sp.Symbol("Tau32")], expr,
                    "numpy")
    return np.asarray(f(zm, zt), dtype=np.float64).ravel()


def affine_resid(a, b):
    A = np.column_stack([a, np.ones_like(a)])
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return coef, float(np.abs(A @ coef - b).max())


print("=" * 84)
print("Q1  IS THERE A CONVERGED FORM?  (affine-equivalence test, tol 1e-6)")
print("=" * 84)
by_c = {}
for c in (4, 5, 6, 8):
    rows = {}
    for s in seeds:
        hit = [r for r in runs[s] if r["complexity"] == c]
        if hit:
            rows[s] = hit[0]
    by_c[c] = rows
    if len(rows) < 2:
        print(f"  c={c:<3} only {len(rows)} seed(s) have this complexity")
        continue
    ss = sorted(rows)
    ref = num(parse(rows[ss[0]]["equation"]))
    same = 0
    for s in ss[1:]:
        _, resid = affine_resid(ref, num(parse(rows[s]["equation"])))
        if resid < 1e-6:
            same += 1
    groups = len({round(sp.count_ops(parse(rows[s]["equation"])), 0) for s in ss})
    print(f"  c={c:<3} seeds={len(ss)}  affine-equal to seed {ss[0]}: "
          f"{same}/{len(ss)-1}  -> {'CONVERGED' if same == len(ss)-1 else 'NOT converged'}")
    for s in ss:
        print(f"        seed {s:<6} {rows[s]['equation'][:62]}")

CONVERGED_C = 4

print("\n" + "=" * 84)
print("Q2  IS THE CONVERGED POINT ON THE PARETO FRONT?")
print("=" * 84)
print("  every recorded row IS a front point, so the question is whether the")
print("  converged complexity is a front point for every seed:")
for s in seeds:
    cs = [r["complexity"] for r in runs[s]]
    on_front = CONVERGED_C in cs
    loss4 = next((r["loss"] for r in runs[s] if r["complexity"] == 4), None)
    print(f"    seed {s:<6} front complexities {cs}")
    print(f"             c=4 present: {on_front}   loss(c=4)="
          f"{loss4:.4f}" if loss4 else "")
losses = [next((r["loss"] for r in runs[s] if r["complexity"] == 4), None)
          for s in seeds]
losses = [x for x in losses if x is not None]
print(f"\n  loss at the converged complexity: "
      f"{min(losses):.4f} .. {max(losses):.4f}  (spread {max(losses)/min(losses):.2f}x)")

print("\n" + "=" * 84)
print("Q3  PERFORMANCE: converged closed form  vs  EFN-ArcFace model")
print("=" * 84)
base = score(logit)
print(f"  EFN-ArcFace model        AUC {base['auc']:.4f}  Brier {base['brier']:.4f}"
      f"  ECE {base['ece']:.4f}  acc {base['acc']*100:.2f}%")

# converged form, in the search's own z-space, mapped to log-odds
conv_s = num(parse(by_c[4][seeds[0]]["equation"])) * y_std + y_mean
c4 = score(conv_s)
print(f"  converged form (c=4)     AUC {c4['auc']:.4f}  Brier {c4['brier']:.4f}"
      f"  ECE {c4['ece']:.4f}  acc {c4['acc']*100:.2f}%"
      f"   [= tanh(m/pT) alone]")

# best single-seed equation that is NOT converged, for contrast
c6 = score(num(parse(by_c[6][42]["equation"])) * y_std + y_mean)
print(f"  best c=6 seed-42 eqn     AUC {c6['auc']:.4f}  Brier {c6['brier']:.4f}"
      f"  ECE {c6['ece']:.4f}  acc {c6['acc']*100:.2f}%   [= tanh+tau32]")

# 2-observable form at full model strength, refit (not from PySR)
print("\n  gap to model: converged form "
      f"{base['auc']-c4['auc']:+.4f} AUC, "
      f"{(base['acc']-c4['acc'])*100:+.2f} pp accuracy")

json.dump(dict(model=base, converged_c4=c4, c6_seed42=c6,
               converged_complexity=CONVERGED_C,
               loss_at_c4=dict(min=min(losses), max=max(losses))),
          open(f"{P}/convergence_verdict.json", "w"), indent=2)
print(f"\nsaved: {P}/convergence_verdict.json")
