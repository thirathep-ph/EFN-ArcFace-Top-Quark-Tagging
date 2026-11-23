"""Default PySR configuration for this project — single source of truth.

Every script that runs PySR on ArcEFN data imports these instead of
hand-rolling its own operator / constraint / normalisation choices.

Design decisions and their reasons
----------------------------------
dims (Buckingham): every dimensionful observable is divided by the jet pT,
    the natural scale of the jet axis the EFN is built on.  sqrt(d12) and
    sqrt(d23) carry GeV (square root of an energy^2 * angle^2 scale), so they
    need /pT just like Mass and mSD.  Buildup:
        Mass/pT, mSD/pT, sqrt_d12/pT, sqrt_d23/pT   [dimensionless]
        Mult, nSD, Tau21, Tau32, zg, theta_g        [already dimensionless]

nested_constraints: ban only compositions that add no expressive power
    (self-repeats and identities such as exp(log(x)) == x).  We deliberately
    KEEP tanh(exp(x)): the empirical seed sweep lands on tanh(exp(...)) in
    4/5 runs, and forcing a full ban makes PySR fall back to a plain linear
    fit (verified on synthetic data).  Binary-mediated nesting is bounded by
    maxdepth, which nested_constraints cannot see.

complexity_of_operators: transcendentals cost more than algebraic operators.
    Operators stay available (this is search, not a ban), but must earn their
    place on the Pareto front.

No math/physics constants are injected.  Constants are fitted freely; if a
fitted constant happens to sit near 1/3, pi/4 etc., that is reported as an
observation, never enforced.
"""
from __future__ import annotations

UNARY_OPERATORS = ["tanh", "exp", "log", "sqrt", "square", "abs"]
BINARY_OPERATORS = ["+", "-", "*", "/", "pow"]

# ── Buckingham-pi feature built-up ─────────────────────────────────────────
# raw column names produced by compute_features(), in order
RAW_FEATURES = ["Mass", "mSD", "Mult", "nSD", "sqrt_d12", "sqrt_d23",
                "Tau21", "Tau32", "zg", "theta_g"]
# which raw features carry energy units and must be divided by pT
DIMENSIONFUL = ["Mass", "mSD", "sqrt_d12", "sqrt_d23"]

DIMLESS_NAMES = ["Mass_over_pT", "mSD_over_pT", "Mult", "nSD",
                 "sqrt_d12_over_pT", "sqrt_d23_over_pT",
                 "Tau21", "Tau32", "zg", "theta_g"]

# ── nested constraint matrix ───────────────────────────────────────────────
# value 0 = composition forbidden anywhere in a subtree
SELF_AND_IDENTITY_BANS = {
    # self-repeat: adds no new shape
    "tanh":   {"tanh": 0},
    "exp":    {"exp": 0},
    "log":    {"log": 0},
    "sqrt":   {"sqrt": 0},
    "square": {"square": 0},
    "abs":    {"abs": 0},
    # identities across operators
    "exp":    {"exp": 0, "log": 0},          # exp(log x) = x
    "log":    {"log": 0, "exp": 0},          # log(exp x) = x
    "sqrt":   {"sqrt": 0, "square": 0},      # sqrt(x^2) = |x|
    "square": {"square": 0, "sqrt": 0},      # (sqrt x)^2 = x
    "abs":    {"abs": 0, "square": 0, "sqrt": 0},  # idempotent / redundant
}
# kept on purpose: tanh(exp(.)), exp(tanh(.)), log(tanh(.)), tanh(log(.))

NESTED_POLICIES = {
    "none": None,
    "self-identity": SELF_AND_IDENTITY_BANS,
}

# ── operator constraints ──────────────────────────────────────────────────
# pow: base may be arbitrarily complex (-1) but the exponent may only be a
# constant or a single variable (1).  Without this PySR can build
# pow(pow(x,y),z) chains that are unreadable and numerically unstable.
CONSTRAINTS = {
    "pow": (-1, 1),
}

# ── complexity weights ────────────────────────────────────────────────────
COMPLEXITY_OF_OPERATORS = {
    "+": 1, "-": 1, "*": 1, "/": 1,
    "tanh": 1, "square": 1, "abs": 1,
    "sqrt": 1.5,
    "pow": 2, "exp": 2, "log": 2,
}

# ── search budget ─────────────────────────────────────────────────────────
BUDGET = {
    "quick": dict(niterations=15, populations=10, population_size=50,
                  maxsize=20, maxdepth=8),
    "full": dict(niterations=30, populations=20, population_size=50,
                 maxsize=20, maxdepth=8),
}

NORMALIZATIONS = {
    # x / sigma -> unit variance but keeps physical zero, so m/pT -> 0 stays 0
    "sigma": "unit-variance, zero-preserving",
    # (x - mean) / sigma -> standard z-score, destroys the physical zero
    "z": "standard z-score",
}


def get_nested_constraints(policy: str):
    if policy not in NESTED_POLICIES:
        raise ValueError(f"unknown nest policy {policy!r}; "
                         f"choose from {sorted(NESTED_POLICIES)}")
    return NESTED_POLICIES[policy]


def get_constraints():
    """Operator arity constraints (pow exponent limited to a constant/variable)."""
    return dict(CONSTRAINTS)


def normalize(X, kind: str):
    """Return (X_norm, block) where block allows exact inversion."""
    import numpy as np
    X = np.asarray(X, dtype=np.float64)
    if kind == "sigma":
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        block = {"norm": "sigma", "x_std": sd.tolist()}
        return X / sd, block
    if kind == "z":
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd[sd == 0] = 1.0
        block = {"norm": "z", "x_mean": mu.tolist(), "x_std": sd.tolist()}
        return (X - mu) / sd, block
    raise ValueError(f"unknown norm {kind!r}; choose from {sorted(NORMALIZATIONS)}")
