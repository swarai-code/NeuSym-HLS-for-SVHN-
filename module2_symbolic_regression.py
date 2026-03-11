
# module2_symbolic_regression.py  –  NeuSym-HLS  |  Module 2
#
# Hardware-Aware Symbolic Regression via PySR
#
# Pipeline
# --------
#   1. Load the layer traces produced by Module 1 from ./traces/.
#   2. For each combination of (distillation level, operator set), run PySR
#      to discover an analytic expression  l = f(h).
#   3. The genetic search is biased toward hardware-efficient functions via
#      custom `complexity_of_operators` weights (clock-cycle costs per op).
#   4. The "Hall of Fame" Pareto front (complexity vs loss) is saved as a CSV
#      and the best equation (Pareto-optimal, complexity ≤ C_max) is pickled.
#
# Outputs  (./sr_results/<level>_<opset>/)
# ----------------------------------------
#   hall_of_fame.csv       – full Pareto front from PySR
#   best_model.pkl         – pickled PySRRegressor
#   best_equation.txt      – human-readable symbolic expression
#   best_equation.sympy    – pickled sympy expression (used by Module 3)
#
# Requirements
# ------------
#   pip install pysr==0.19.*     (PySR ≥ 0.19 uses the v1.5.0 Julia backend)
#   Julia runtime must be installed and PySR.install() must have been run once.


import os
import pathlib
import pickle
import textwrap
import numpy as np

# PySR ≥ 0.19 re-exports PySRRegressor from pysr directly
try:
    from pysr import PySRRegressor
except ImportError as exc:
    raise ImportError(
        "PySR is not installed.  Run:  pip install pysr==0.19.*\n"
        "Then launch Python once and call:  import pysr; pysr.install()"
    ) from exc

# Local utilities (hardware cost table & operator-set definitions)
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils.hw_costs import get_operator_set, OPERATOR_SETS

#Paths
ROOT       = pathlib.Path(__file__).parent
TRACE_DIR  = ROOT / "traces"
SR_DIR     = ROOT / "sr_results"
SR_DIR.mkdir(parents=True, exist_ok=True)

#PySR search hyper-parameters (fixed per paper specification)
NITERATIONS = 40
POPULATIONS  = 10

# Hardware complexity budget: equations whose total operator-cost exceeds this
# threshold are not selected as the "best" model even if they fit well.
# Empirically: C_max = 30 allows polynomials of degree ≤ 4 in a few variables.
C_MAX = 30

# L2 regression loss (Julia-side expression string for PySR)
L2_LOSS = "loss(prediction, target) = (prediction - target)^2"

#Distillation levels
# Maps a label to the hidden-vector npy file and its dimensionality.
DISTILL_LEVELS = {
    "1L": {"hidden_file": "1L_hidden.npy", "dim": 128},
    "2L": {"hidden_file": "2L_hidden.npy", "dim": 512},
}



# Helpers

def _load_traces(level: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load hidden activations H and target logits L from ./traces/.

    Returns
    -------
    H : np.ndarray  shape (N, D)   – hidden vectors
    L : np.ndarray  shape (N,)     – scalar logits  (PySR expects 1-D target)
    """
    cfg = DISTILL_LEVELS[level]
    H   = np.load(TRACE_DIR / cfg["hidden_file"])            # (N, D)
    L   = np.load(TRACE_DIR / "logits.npy").squeeze(-1)      # (N,)
    assert H.shape[0] == L.shape[0], "Trace length mismatch"
    print(f"[Module 2] Loaded traces for {level}:  H {H.shape}, L {L.shape}")
    return H, L


def _compute_equation_hw_cost(model: PySRRegressor, c_max: int = C_MAX) -> dict:
    """
    Walk PySR's hall-of-fame equations and tag each with its total hardware
    latency cost.  Returns the best equation whose cost ≤ c_max.

    The 'complexity' column in PySR's equations DataFrame already encodes a
    node-count metric; when we supply `complexity_of_operators`, PySR scales
    each node's contribution by the assigned cost.  This function re-computes
    the metric for transparency and applies the C_max budget filter.
    """
    df = model.equations_
    if df is None or df.empty:
        return {}

    # PySR's 'complexity' column reflects the weighted sum when custom costs
    # are provided – so we use it directly as the hardware proxy.
    hw_costs = df["complexity"].values
    losses   = df["loss"].values

    # Filter by budget
    feasible = df[hw_costs <= c_max].copy()
    if feasible.empty:
        print(f"[Module 2] WARNING: No equation within C_max={c_max}. "
              "Relaxing to lowest-complexity equation.")
        feasible = df.sort_values("complexity").head(1)

    # Among feasible equations, pick lowest loss
    best_row = feasible.loc[feasible["loss"].idxmin()]
    return {
        "equation":   best_row["equation"],
        "sympy_form": best_row.get("sympy_format", best_row["equation"]),
        "complexity": int(best_row["complexity"]),
        "loss":       float(best_row["loss"]),
    }



# Core: run one PySR experiment

def run_sr(level:   str,
           opset:   str,
           c_max:   int  = C_MAX,
           seed:    int  = 42,
           verbose: bool = True) -> dict:
    """
    Run PySR symbolic regression for one (level, opset) combination.

    Parameters
    ----------
    level   : "1L" or "2L"
    opset   : "SCE", "SRL", or "POL"
    c_max   : maximum hardware-weighted complexity budget
    seed    : random seed passed to PySR
    verbose : pass through to PySR

    Returns
    -------
    result : dict with keys
        'equation'    – string representation of the best equation
        'sympy_expr'  – sympy expression object (for Module 3 code-gen)
        'complexity'  – hardware-weighted complexity score
        'loss'        – MSE on training traces
        'model'       – fitted PySRRegressor object
    """
    H, L  = _load_traces(level)
    op_cfg = get_operator_set(opset)

    # Output directory
    out_dir = SR_DIR / f"{level}_{opset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Module 2] ── SR run: level={level}, opset={opset} ──────────────")
    print(f"           Operators: {op_cfg}")
    print(f"           niterations={NITERATIONS}, populations={POPULATIONS}, "
          f"C_max={c_max}\n")

    #PySR model construction
    # PySR v1.5.0 API (Python package ≥ 0.19)
    # Key knobs for hardware awareness
    #   complexity_of_operators  : dict mapping op → cycle-cost integer
    #   maxsize                  : hard cap on expression node count
    #   parsimony                : additional regularisation on complexity
    model = PySRRegressor(
        #Search budget 
        niterations  = NITERATIONS,
        populations  = POPULATIONS,
        population_size = 50,           # individuals per island
        ncycles_per_iteration = 550,    # genetic-search cycles per iteration

        # Operator sets (from hw_costs.py)
        binary_operators = op_cfg["binary_operators"],
        unary_operators  = op_cfg.get("unary_operators", []),

        # Hardware-aware complexity biasing 
        # Each node's contribution to complexity is multiplied by its cost.
        # This causes the Pareto front to prefer low-cycle-count expressions.
        complexity_of_operators = op_cfg["complexity_of_operators"],
        # Additional per-node base complexity penalty (keep constant at 1)
        complexity_of_constants = 1,
        complexity_of_variables = 1,
        # Hard node-count ceiling independent of cost weighting
        maxsize   = 40,
        # Parsimony coefficient: penalises complexity in the fitness score
        # λ_parsimony: higher → simpler expressions preferred
        parsimony = 0.0032,

        # Regression objective
        loss = L2_LOSS,

        #Output and reproducibility
        output_jl_filename = str(out_dir / "sr_search.jl"),
        temp_equation_file = str(out_dir / "hall_of_fame.csv"),
        verbosity = int(verbose),
        random_state = seed,
        deterministic  = True,         # reproducible Julia RNG

        #Performance 
        procs = 0,                     # 0 = use all available Julia threads
        multithreading = True,
    )

    #Fit
    # X must be 2-D array of shape (N, D); y is 1-D (N,).
    model.fit(H, L)

    #Select best hardware-feasible equation 
    best = _compute_equation_hw_cost(model, c_max=c_max)
    if not best:
        raise RuntimeError("PySR returned an empty hall of fame – check Julia installation.")

    print(f"\n[Module 2] Best equation (within C_max={c_max}):")
    print(f"           {best['equation']}")
    print(f"           Complexity={best['complexity']}, MSE={best['loss']:.6f}\n")

    #Retrieve sympy expression
    # PySR stores the sympy expression for the best equation via .sympy()
    # We use the overall best (lowest loss) from the Pareto front.
    try:
        sympy_expr = model.sympy()
    except Exception:
        # Fallback: convert equation string to sympy manually
        import sympy
        sympy_expr = sympy.sympify(best["sympy_form"])

    #Persist results 
    # Hall of fame CSV (already written by PySR via temp_equation_file)
    hof_csv = out_dir / "hall_of_fame.csv"
    if model.equations_ is not None:
        model.equations_.to_csv(hof_csv, index=False)
        print(f"[Module 2] Hall of fame → {hof_csv}")

    # Pickled model
    pkl_path = out_dir / "best_model.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    print(f"[Module 2] PySRRegressor  → {pkl_path}")

    # Human-readable equation
    eq_txt = out_dir / "best_equation.txt"
    eq_txt.write_text(textwrap.dedent(f"""
        Level     : {level}
        Opset     : {opset}
        Equation  : {best['equation']}
        Complexity: {best['complexity']} (C_max={c_max})
        MSE       : {best['loss']:.8f}
    """).strip())
    print(f"[Module 2] Equation text  → {eq_txt}")

    # Sympy expression (used by Module 3 for HLS code generation)
    sym_path = out_dir / "best_equation.sympy.pkl"
    with open(sym_path, "wb") as f:
        pickle.dump(sympy_expr, f)
    print(f"[Module 2] Sympy expr     → {sym_path}")

    return {
        "equation":   best["equation"],
        "sympy_expr": sympy_expr,
        "complexity": best["complexity"],
        "loss":       best["loss"],
        "model":      model,
        "out_dir":    out_dir,
    }



# Sweep: run all (level × opset) combinations

def run_sweep(levels: list[str] | None   = None,
              opsets: list[str] | None   = None,
              c_max:  int                = C_MAX) -> dict[str, dict]:
    """
    Run the full NeuSym-HLS SR sweep and return a nested results dictionary.

        results[(level, opset)] = { 'equation', 'sympy_expr', 'complexity',
                                    'loss', 'model', 'out_dir' }
    """
    if levels is None:
        levels = list(DISTILL_LEVELS.keys())           # ["1L", "2L"]
    if opsets is None:
        opsets = list(OPERATOR_SETS.keys())            # ["SCE", "SRL", "POL"]

    results = {}
    for lvl in levels:
        for ops in opsets:
            key = (lvl, ops)
            try:
                results[key] = run_sr(lvl, ops, c_max=c_max)
            except Exception as exc:
                print(f"[Module 2] ERROR for {key}: {exc}")
                results[key] = {"error": str(exc)}

    #Summary table 
    print("\n[Module 2] ══ Sweep Summary ══════════════════════════════════════")
    print(f"{'Config':<12} {'Complexity':>12} {'MSE':>14}")
    print("─" * 42)
    for (lvl, ops), res in results.items():
        if "error" in res:
            print(f"{lvl}-{ops:<8} {'ERROR':>12} {res['error'][:20]:>14}")
        else:
            print(f"{lvl}-{ops:<8} {res['complexity']:>12} {res['loss']:>14.6f}")
    print()

    return results



# Entry point

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeuSym-HLS Module 2 – Symbolic Regression")
    parser.add_argument("--level",  choices=["1L", "2L", "both"],  default="1L",
                        help="Distillation level (default: 1L → SR-1L-POL target)")
    parser.add_argument("--opset",  choices=["SCE", "SRL", "POL", "all"], default="POL",
                        help="Operator set (default: POL → paper's best result)")
    parser.add_argument("--c_max",  type=int, default=C_MAX,
                        help=f"Hardware complexity budget (default: {C_MAX})")
    args = parser.parse_args()

    levels = ["1L", "2L"] if args.level == "both" else [args.level]
    opsets = ["SCE", "SRL", "POL"] if args.opset == "all"  else [args.opset]

    run_sweep(levels=levels, opsets=opsets, c_max=args.c_max)
