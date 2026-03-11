# =============================================================================
# utils/hw_costs.py
#
# Hardware-aware operator cost table for the AMD XC7Z020 FPGA @ 100 MHz.
#
# Latency estimates are in clock cycles at 100 MHz for single-precision float
# operations, derived from Xilinx floating-point IP core datasheets and
# Vitis HLS synthesis reports. These costs are fed to PySR as
# `complexity_of_operators` to penalise hardware-heavy functions and steer
# the genetic search toward LUT/DSP-efficient analytic expressions.
#
# Reference tiers
# ---------------
#   Cost 1  : trivial (wire / adder)
#   Cost 2  : single DSP slice (multiplier)
#   Cost 4  : small CORDIC / LUT-only approximation
#   Cost 8  : medium CORDIC (sin/cos ≈ 8-stage pipeline)
#   Cost 12 : exp / log via Taylor + normalisation
#   Cost 16 : division (iterative reciprocal)
# =============================================================================

# ── Per-operator latency costs (used by PySR as complexity_of_operators) ──────

OPERATOR_COSTS: dict[str, int] = {
    # Binary operators
    "+":  1,   # float adder          – 1 DSP or pure LUT
    "-":  1,   # float subtractor     – same as adder
    "*":  2,   # float multiplier     – 1 DSP @ 2-3 cycles
    "/": 16,   # float divider        – iterative Newton-Raphson, expensive
    # Unary operators
    "sin":    8,   # CORDIC sin           – 8-stage pipeline
    "cos":    8,   # CORDIC cos           – 8-stage pipeline
    "exp":   12,   # exp via range-redux + polynomial
    "log":   12,   # log via range-redux + polynomial
    "square": 2,   # x*x  (one multiplier)
    "relu":   1,   # max(0,x) – comparator only
    "abs":    1,   # sign-bit flip
    "neg":    1,   # sign-bit flip
    "sqrt":  10,   # iterative Newton-Raphson
    "tanh":  14,   # CORDIC or LUT table
    "cube":   4,   # x*x*x (two multipliers)
}

# ── Canonical operator sets referenced in the paper ──────────────────────────

OPERATOR_SETS: dict[str, dict] = {
    # ── SCE : Sin-Cos-Exp (richest; most expensive on FPGA) ──────────────────
    "SCE": {
        "binary_operators": ["+", "-", "*"],
        "unary_operators":  ["sin", "cos", "exp"],
        "complexity_of_operators": {
            "+": OPERATOR_COSTS["+"],
            "-": OPERATOR_COSTS["-"],
            "*": OPERATOR_COSTS["*"],
            "sin": OPERATOR_COSTS["sin"],
            "cos": OPERATOR_COSTS["cos"],
            "exp": OPERATOR_COSTS["exp"],
        },
    },
    # ── SRL : Square-ReLU (hardware-friendly nonlinearity) ───────────────────
    "SRL": {
        "binary_operators": ["+", "-", "*"],
        "unary_operators":  ["square", "relu"],
        "complexity_of_operators": {
            "+":      OPERATOR_COSTS["+"],
            "-":      OPERATOR_COSTS["-"],
            "*":      OPERATOR_COSTS["*"],
            "square": OPERATOR_COSTS["square"],
            "relu":   OPERATOR_COSTS["relu"],
        },
    },
    # ── POL : Polynomial (cheapest; pure DSP-adder chain) ────────────────────
    "POL": {
        "binary_operators": ["+", "-", "*", "/"],
        "unary_operators":  [],
        "complexity_of_operators": {
            "+": OPERATOR_COSTS["+"],
            "-": OPERATOR_COSTS["-"],
            "*": OPERATOR_COSTS["*"],
            "/": OPERATOR_COSTS["/"],
        },
    },
}


def get_operator_set(name: str) -> dict:
    """Return the PySR operator-set configuration dict for *name* ∈ {SCE, SRL, POL}."""
    name = name.upper()
    if name not in OPERATOR_SETS:
        raise ValueError(f"Unknown operator set '{name}'. Choose from {list(OPERATOR_SETS)}")
    return OPERATOR_SETS[name]
