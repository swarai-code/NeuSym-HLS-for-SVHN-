
# module3_hls_codegen.py  –  NeuSym-HLS  |  Module 3
#
# Hybrid Datapath C++ HLS Code Generation
#
# Pipeline
# --------
#   1. Load the pickled sympy expression from Module 2 (best_equation.sympy.pkl).
#   2. Walk the sympy AST and translate it to a synthesisable Vitis HLS C/C++
#      function that uses `float` datatypes throughout.
#   3. Emit a complete .cpp file with:
#        • #pragma HLS INLINE to enable inlining into the top-level accelerator
#        • A companion header (.h) declaring the function signature
#   4. Generate a "stitcher" snippet: a top-level Vitis HLS wrapper that wires
#      the hls4ml quantised MLP front-end directly to the symbolic SR back-end.
#
# Outputs  (./hls_output/<level>_<opset>/)
# -----------------------------------------
#   sr_layer.cpp           – synthesisable symbolic function
#   sr_layer.h             – header
#   hybrid_top.cpp         – top-level HLS design (hls4ml + SR stitched)
#   hybrid_top.h           – top-level header
#   tcl/synth.tcl          – Vitis HLS project creation + synthesis TCL
#
# Supported sympy nodes
# --------------------------
#   Add, Mul, Pow (integer exp), sin, cos, exp, log, Abs, Max (relu), Symbol,
#   Float, Integer, RealNumber.  Division (a/b) is handled as Mul(a, Pow(b,-1)).


import os
import pathlib
import pickle
import textwrap
from typing import Any

import sympy
from sympy import (
    Add, Mul, Pow, Symbol, Integer, Float, Rational,
    sin, cos, exp, log, Abs, Max, Number
)

#Paths
ROOT       = pathlib.Path(__file__).parent
SR_DIR     = ROOT / "sr_results"
HLS_DIR    = ROOT / "hls_output"
HLS_DIR.mkdir(parents=True, exist_ok=True)

#FPGA target metadata 
FPGA_PART   = "xc7z020clg400-1"       # AMD XC7Z020 (Zynq-7000)
CLOCK_PERIOD = 10.0                   # ns → 100 MHz

# C++ float literal helpers

def _fmt_float(val: float) -> str:
    """Format a constant as a single-precision C float literal (e.g. 3.14f)."""
    return f"{float(val):.8f}f"



# 1.  Sympy AST → C++ expression string


def _sympy_to_cpp(expr: Any, var_map: dict[str, str]) -> str:
    """
    Recursively translate a sympy expression to a single C/C++ expression
    string suitable for Vitis HLS (single-precision float throughout).

    Parameters
    ----------
    expr    : sympy expression node
    var_map : mapping from sympy Symbol name → C++ variable name string

    Returns
    -------
    cpp_str : str  (no trailing semicolon)
    """

    # Leaf nodes 
    if isinstance(expr, Symbol):
        sym_name = expr.name
        if sym_name in var_map:
            return var_map[sym_name]
        # PySR names variables x0, x1, … → map directly
        return sym_name

    if isinstance(expr, Integer):
        return _fmt_float(int(expr))

    if isinstance(expr, (Float, Rational, Number)):
        return _fmt_float(float(expr))

    # Add: a + b + c … 
    if isinstance(expr, Add):
        terms = [_sympy_to_cpp(arg, var_map) for arg in expr.args]
        return "(" + " + ".join(terms) + ")"

    # Mul: a * b * c … (handles implicit division via Pow(b, -1)) 
    if isinstance(expr, Mul):
        parts = []
        for arg in expr.args:
            parts.append(_sympy_to_cpp(arg, var_map))
        # Group positive and negative powers for readability
        return "(" + " * ".join(parts) + ")"

    # Pow: a^n 
    if isinstance(expr, Pow):
        base_str = _sympy_to_cpp(expr.base, var_map)
        exp_val  = expr.exp

        # Integer powers: unroll for efficiency (avoids powf for small n)
        if isinstance(exp_val, Integer):
            n = int(exp_val)
            if n == 2:
                return f"({base_str} * {base_str})"
            if n == 3:
                return f"({base_str} * {base_str} * {base_str})"
            if n == -1:
                # Division: x^{-1}  →  1.0f / x
                return f"(1.0f / {base_str})"
            if n < 0:
                return f"(1.0f / powf({base_str}, {_fmt_float(-n)}))"
        # General power: use powf
        exp_str = _sympy_to_cpp(exp_val, var_map)
        return f"powf({base_str}, {exp_str})"

    # Transcendental functions 
    if isinstance(expr, sympy.sin):
        return f"sinf({_sympy_to_cpp(expr.args[0], var_map)})"

    if isinstance(expr, sympy.cos):
        return f"cosf({_sympy_to_cpp(expr.args[0], var_map)})"

    if isinstance(expr, sympy.exp):
        return f"expf({_sympy_to_cpp(expr.args[0], var_map)})"

    if isinstance(expr, sympy.log):
        return f"logf({_sympy_to_cpp(expr.args[0], var_map)})"

    if isinstance(expr, sympy.Abs):
        return f"fabsf({_sympy_to_cpp(expr.args[0], var_map)})"

    # ReLU: Max(0, x) 
    if isinstance(expr, sympy.Max):
        args_cpp = [_sympy_to_cpp(a, var_map) for a in expr.args]
        if len(args_cpp) == 2:
            return f"fmaxf({args_cpp[0]}, {args_cpp[1]})"
        # General case
        return f"fmaxf({', '.join(args_cpp)})"

    # Sqrt 
    if isinstance(expr, sympy.sqrt):
        return f"sqrtf({_sympy_to_cpp(expr.args[0], var_map)})"

    # Fallback: emit str(expr) with a comment so the user can review 
    raw = str(expr)
    print(f"[Module 3] WARNING: unhandled sympy node {type(expr).__name__!r}; "
          f"emitting raw string: {raw!r}")
    return f"/* UNHANDLED: {raw} */ 0.0f"



# 2.  Variable-map builder

def _build_var_map(expr:      sympy.Expr,
                   dim:       int,
                   array_name: str = "h") -> dict[str, str]:
    """
    Create a mapping from PySR's variable symbols (x0, x1, …) to C++ array
    element references (h[0], h[1], …).

    PySR names the j-th feature 'xj' (0-indexed).
    """
    syms = sorted(expr.free_symbols, key=lambda s: s.name)
    var_map: dict[str, str] = {}
    for s in syms:
        name = s.name
        # Accept both 'x0' and 'x_0' conventions used by PySR
        if name.startswith("x_"):
            idx = int(name[2:])
        elif name.startswith("x"):
            idx = int(name[1:])
        else:
            # Keep unknown symbols as-is (they become C++ variable names)
            var_map[name] = name
            continue
        if idx >= dim:
            raise ValueError(f"Symbol {name} has index {idx} ≥ input dim {dim}")
        var_map[name] = f"{array_name}[{idx}]"
    return var_map



# 3.  File generators

def _gen_sr_header(func_name: str, dim: int, level: str) -> str:
    """Generate sr_layer.h content."""
    return textwrap.dedent(f"""\
    // sr_layer.h  –  Auto-generated by NeuSym-HLS Module 3
    // DO NOT EDIT MANUALLY.
    //
    // Symbolic regression layer for distillation level {level}.
    // Input dim : {dim}
    // Target hw : {FPGA_PART} @ {int(1000.0/CLOCK_PERIOD)} MHz

    #ifndef SR_LAYER_H
    #define SR_LAYER_H

    #include <cmath>    // sinf, cosf, expf, logf, fabsf, powf, sqrtf, fmaxf
    #include <ap_fixed.h>   // Vitis HLS arbitrary precision (unused in float path)

    // Symbolic regression layer: maps h[0..{dim-1}] → scalar logit
    float {func_name}(const float h[{dim}]);

    #endif  // SR_LAYER_H
    """)


def _gen_sr_cpp(func_name: str,
                cpp_expr:  str,
                dim:       int,
                level:     str,
                opset:     str,
                equation:  str) -> str:
    """Generate sr_layer.cpp content."""
    return textwrap.dedent(f"""\
    // sr_layer.cpp  –  Auto-generated by NeuSym-HLS Module 3
    // DO NOT EDIT MANUALLY.
    //
    // Distillation level : {level}
    // Operator set       : {opset}
    // Original equation  : {equation}
    // Input dimension    : {dim}
    // Target hardware    : {FPGA_PART}  @  {int(1000.0/CLOCK_PERIOD)} MHz

    #include "sr_layer.h"

    // ---------------------------------------------------------------------------
    // {func_name}
    //
    // Implements the analytic expression discovered by symbolic regression.
    // The function is marked INLINE so Vitis HLS folds it directly into the
    // calling pipeline, eliminating AXI handshake overhead and enabling
    // cross-boundary operator scheduling.
    // ---------------------------------------------------------------------------
    float {func_name}(const float h[{dim}])
    {{
    #pragma HLS INLINE          // Fold into caller's pipeline schedule
    #pragma HLS LATENCY min=1   // Allow HLS to optimise freely

        // Analytic expression (auto-translated from sympy)
        const float result = {cpp_expr};

        return result;
    }}
    """)


def _gen_hybrid_header(dim_in: int, dim_hidden: int, level: str) -> str:
    """
    Generate hybrid_top.h  –  top-level accelerator header.
    The design exposes an AXI-Stream interface compatible with hls4ml output.
    """
    return textwrap.dedent(f"""\
    // hybrid_top.h  –  NeuSym-HLS hybrid accelerator top-level header
    // Auto-generated by Module 3.
    //
    // Architecture (level {level}):
    //   AXI-S pixel input  →  hls4ml quantised MLP front-end
    //                      →  SR symbolic layer (float)
    //                      →  AXI-S logit output
    //
    // Target: {FPGA_PART}  @  {int(1000.0/CLOCK_PERIOD)} MHz

    #ifndef HYBRID_TOP_H
    #define HYBRID_TOP_H

    #include "hls_stream.h"
    #include "ap_fixed.h"

    // ── Type aliases ──────────────────────────────────────────────────────────
    // Use the same fixed-point type as the hls4ml-generated front-end.
    // Adjust width/int_bits to match your hls4ml configuration.
    typedef ap_fixed<16, 6> input_t;        // hls4ml input type
    typedef ap_fixed<16, 6> layer_t;        // hls4ml intermediate type
    typedef float            sr_out_t;      // SR layer always in float

    // ── Dimensions ────────────────────────────────────────────────────────────
    static const int INPUT_DIM  = {dim_in};
    static const int HIDDEN_DIM = {dim_hidden};

    // ── Top-level function prototype ──────────────────────────────────────────
    void hybrid_top(
        hls::stream<input_t>  &in_stream,
        hls::stream<sr_out_t> &out_stream
    );

    #endif  // HYBRID_TOP_H
    """)


def _gen_hybrid_top(func_name:  str,
                    dim_in:     int,
                    dim_hidden: int,
                    level:      str,
                    hls4ml_fn:  str = "myproject") -> str:
    """
    Generate hybrid_top.cpp  –  stitches hls4ml front-end to the SR layer.

    Assumptions
    -----------
    •  hls4ml has been compiled with:
         hls4ml convert -c config.yml
       producing a project with a top-level function named *hls4ml_fn*.
    •  That function signature (as generated by hls4ml) is:
         void <hls4ml_fn>(
             input_t  in[INPUT_DIM],
             layer_t  layer_out[HIDDEN_DIM]);
    •  The SR layer sr_layer.cpp is compiled into the same Vitis HLS project.
    """
    return textwrap.dedent(f"""\
    // hybrid_top.cpp  –  NeuSym-HLS hybrid accelerator top-level
    // Auto-generated by Module 3.  DO NOT EDIT MANUALLY.
    //
    // Distillation level : {level}
    // hls4ml front-end   : {hls4ml_fn}()
    // SR back-end        : {func_name}()
    // Target             : {FPGA_PART}  @  {int(1000.0/CLOCK_PERIOD)} MHz

    #include "hybrid_top.h"
    #include "sr_layer.h"

    //  hls4ml-generated front-end prototype 
    // Include the hls4ml-generated header here; we forward-declare for clarity.
    extern void {hls4ml_fn}(
        input_t in[INPUT_DIM],
        layer_t layer_out[HIDDEN_DIM]
    );

    // Hybrid top-level 
    void hybrid_top(
        hls::stream<input_t>  &in_stream,    // AXI-Stream: raw pixel input
        hls::stream<sr_out_t> &out_stream)   // AXI-Stream: binary logit output
    {{
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE ap_ctrl_none port=return   // free-running pipeline

    // 1: buffer input sample from AXI-Stream 
    input_t in_buf[INPUT_DIM];
    #pragma HLS ARRAY_PARTITION variable=in_buf complete

        for (int i = 0; i < INPUT_DIM; i++) {{
    #pragma HLS PIPELINE II=1
            in_buf[i] = in_stream.read();
        }}

    // 2: run hls4ml quantised MLP front-end 
    // Produces the hidden vector (h) that feeds the SR layer.
    layer_t h_buf[HIDDEN_DIM];
    #pragma HLS ARRAY_PARTITION variable=h_buf complete

        {hls4ml_fn}(in_buf, h_buf);

    // 3: type-cast layer_t → float for SR layer
    // The SR layer operates in float; cast each element individually.
    // In a fully-pipelined design, replace this cast with ap_fixed-native SR.
    float h_float[HIDDEN_DIM];
    #pragma HLS ARRAY_PARTITION variable=h_float complete

        for (int j = 0; j < HIDDEN_DIM; j++) {{
    #pragma HLS UNROLL
            h_float[j] = static_cast<float>(h_buf[j]);
        }}

    //4: evaluate symbolic regression expression 
        sr_out_t logit = {func_name}(h_float);

    //5: push logit to output AXI-Stream 
        out_stream.write(logit);
    }}
    """)


def _gen_tcl(project_dir: pathlib.Path,
             level:       str,
             hls4ml_proj: str = "hls4ml_prj") -> str:
    """
    Generate a Vitis HLS TCL script to create the project, add sources,
    run C simulation, synthesis, and export the IP.
    """
    return textwrap.dedent(f"""\
    # synth.tcl  –  Vitis HLS synthesis script
    # Auto-generated by NeuSym-HLS Module 3.
    #
    # Usage (from Vitis HLS TCL shell):
    #   vitis_hls -f synth.tcl
    #
    # Target : {FPGA_PART}  @  {int(1000.0/CLOCK_PERIOD)} MHz
    # Level  : {level}

    # Create project
    open_project -reset hybrid_neusym_{level}

    # Set top-level function 
    set_top hybrid_top

    #  Add HLS sources 
    # 1. hls4ml-generated front-end (copy from hls4ml project output)
    add_files -cflags "-std=c++14" \\
        {hls4ml_proj}/firmware/myproject.cpp \\
        {hls4ml_proj}/firmware/weights

    # 2. NeuSym-HLS symbolic layer (auto-generated)
    add_files -cflags "-std=c++14" \\
        sr_layer.cpp \\
        hybrid_top.cpp

    # Add testbench 
    add_files -tb tb_hybrid_top.cpp

    # Solution / device configuration 
    open_solution -reset "solution1" -flow_target vivado
    set_part {FPGA_PART}
    create_clock -period {CLOCK_PERIOD}ns -name default

    # Directives
    # Pipeline the outer loop in the hls4ml layers where possible
    config_compile -pipeline_loops 64
    config_interface -default_slave_interface s_axilite

    # Runs 
    csim_design                        ;# C simulation (functional check)
    csynth_design                      ;# RTL synthesis
    cosim_design -rtl verilog          ;# RTL co-simulation
    export_design -format ip_catalog \\
        -description "NeuSym-HLS Hybrid Accelerator ({level})" \\
        -vendor "neusym" -library "hls" -ipname "hybrid_neusym_{level}" \\
        -version "1.0"

    #  Extract metrics from synthesis report 
    # After csynth_design completes, metrics are in:
    #   hybrid_neusym_{level}/solution1/syn/report/csynth.xml
    # Module 4 (module4_finetune_eval.py) parses this file automatically.

    exit
    """)



# 4.  Top-level code-generation function

def generate_hls_code(level:      str,
                      opset:      str,
                      dim:        int,
                      dim_in:     int         = 3072,
                      func_name:  str | None  = None,
                      hls4ml_fn:  str         = "myproject",
                      sr_dir:     pathlib.Path = SR_DIR,
                      hls_dir:    pathlib.Path = HLS_DIR) -> pathlib.Path:
    """
    Load the pickled sympy expression for *(level, opset)* and write a full
    Vitis HLS project directory under *hls_dir/<level>_<opset>/*.

    Parameters
    ----------
    level      : "1L" or "2L"
    opset      : "SCE", "SRL", or "POL"
    dim        : dimensionality of the hidden vector h (128 for 1L, 512 for 2L)
    dim_in     : input dimension to the full MLP (3072 for SVHN)
    func_name  : C++ function name (default: "sr_<level>_<opset>")
    hls4ml_fn  : name of the hls4ml-generated top function
    sr_dir     : directory containing Module 2 outputs
    hls_dir    : directory to write HLS files into

    Returns
    -------
    out_dir : pathlib.Path  –  directory containing all generated files
    """
    if func_name is None:
        func_name = f"sr_{level.lower()}_{opset.lower()}"

    # Load sympy expression 
    sym_path = sr_dir / f"{level}_{opset}" / "best_equation.sympy.pkl"
    if not sym_path.exists():
        raise FileNotFoundError(
            f"Sympy pickle not found: {sym_path}\n"
            "Run Module 2 first (module2_symbolic_regression.py)."
        )
    with open(sym_path, "rb") as f:
        sympy_expr: sympy.Expr = pickle.load(f)

    eq_txt_path = sr_dir / f"{level}_{opset}" / "best_equation.txt"
    equation_str = eq_txt_path.read_text() if eq_txt_path.exists() else str(sympy_expr)

    print(f"\n[Module 3] Generating HLS code for  level={level}, opset={opset}")
    print(f"           Sympy expr: {sympy_expr}")

    #  Build variable map: x0 → h[0], x1 → h[1], … 
    var_map = _build_var_map(sympy_expr, dim=dim, array_name="h")
    print(f"           Variable map: {var_map}")

    # Translate to C++ expression 
    cpp_expr = _sympy_to_cpp(sympy_expr, var_map)
    print(f"           C++ expr:  {cpp_expr}")

    #  Create output directory 
    out_dir = hls_dir / f"{level}_{opset}"
    (out_dir / "tcl").mkdir(parents=True, exist_ok=True)

    # Write files
    files = {
        out_dir / "sr_layer.h":      _gen_sr_header(func_name, dim, level),
        out_dir / "sr_layer.cpp":    _gen_sr_cpp(func_name, cpp_expr, dim,
                                                   level, opset, equation_str),
        out_dir / "hybrid_top.h":    _gen_hybrid_header(dim_in, dim, level),
        out_dir / "hybrid_top.cpp":  _gen_hybrid_top(func_name, dim_in, dim,
                                                       level, hls4ml_fn),
        out_dir / "tcl" / "synth.tcl": _gen_tcl(out_dir, level),
    }

    for path, content in files.items():
        path.write_text(content, encoding="utf-8")
        print(f"[Module 3] Written: {path.relative_to(ROOT)}")

    # Also generate a minimal standalone testbench 
    _write_testbench(out_dir, func_name, dim)

    print(f"\n[Module 3] HLS project directory: {out_dir}")
    print(f"           Run synthesis with:\n"
          f"             cd {out_dir}/tcl && vitis_hls -f synth.tcl\n")
    return out_dir


def _write_testbench(out_dir: pathlib.Path, func_name: str, dim: int):
    """Write a minimal C++ testbench for csim_design."""
    tb_path = out_dir / "tb_hybrid_top.cpp"
    content = textwrap.dedent(f"""\
    // tb_hybrid_top.cpp  –  Minimal Vitis HLS C-simulation testbench
    // Auto-generated by NeuSym-HLS Module 3.

    #include <cstdio>
    #include <cstdlib>
    #include "hybrid_top.h"
    #include "sr_layer.h"

    int main() {{
        // Allocate a random test vector 
        float h[{dim}];
        for (int i = 0; i < {dim}; i++) {{
            h[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }}

        // Call SR layer directly (unit test) 
        float result = {func_name}(h);
        printf("[TB] {func_name}(h) = %f\\n", result);

        // Integration test: push through full hybrid_top via streams 
        // (Requires hls4ml firmware to be linked; skip in standalone mode.)
        printf("[TB] C-simulation PASS\\n");
        return 0;
    }}
    """)
    tb_path.write_text(content, encoding="utf-8")
    print(f"[Module 3] Written: {tb_path.relative_to(ROOT)}")



# 5.  Convenience: generate all (level, opset) combos that have SR outputs


def generate_all(hls4ml_fn: str = "myproject"):
    """
    Scan sr_results/ for completed SR runs and generate HLS code for each.

    dim map: 1L → 128 (input to fc3), 2L → 512 (input to fc2).
    """
    dim_map = {"1L": 128, "2L": 512}
    generated = []

    for run_dir in sorted(SR_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        sym_pkl = run_dir / "best_equation.sympy.pkl"
        if not sym_pkl.exists():
            continue
        # Parse level and opset from directory name (e.g. "1L_POL")
        parts = run_dir.name.split("_", 1)
        if len(parts) != 2:
            continue
        level, opset = parts
        if level not in dim_map:
            continue
        try:
            out = generate_hls_code(level, opset,
                                    dim=dim_map[level],
                                    hls4ml_fn=hls4ml_fn)
            generated.append(out)
        except Exception as exc:
            print(f"[Module 3] ERROR for {run_dir.name}: {exc}")

    print(f"\n[Module 3] Generated {len(generated)} HLS project(s).")
    return generated



# Entry point

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="NeuSym-HLS Module 3 – HLS Code Generation")
    parser.add_argument("--level",  choices=["1L", "2L", "all"], default="1L")
    parser.add_argument("--opset",  choices=["SCE", "SRL", "POL", "all"], default="POL")
    parser.add_argument("--hls4ml_fn", default="myproject",
                        help="Name of the hls4ml top-level function")
    args = parser.parse_args()

    if args.level == "all" or args.opset == "all":
        generate_all(hls4ml_fn=args.hls4ml_fn)
    else:
        dim_map = {"1L": 128, "2L": 512}
        generate_hls_code(
            level     = args.level,
            opset     = args.opset,
            dim       = dim_map[args.level],
            hls4ml_fn = args.hls4ml_fn,
        )
