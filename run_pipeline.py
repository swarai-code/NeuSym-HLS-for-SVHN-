
# run_pipeline.py  -  NeuSym-HLS  Top-Level Orchestrator
#
# Executes all five modules in sequence:
#   Module 1  ->  Train FP32 MLP, save baseline accuracy, extract traces
#   Module 2  ->  Symbolic regression (PySR), save hall_of_fame.csv
#   Module 3  ->  Generate Vitis HLS C++ for the hybrid datapath
#   Module 4  ->  Fine-tune hybrid model, save accuracy_log.csv
#                 (optionally parse Vitis HLS synthesis report)
#   Module 5  ->  Read all result files, generate evaluation charts
#
# Module 5 is always the last step and reads ONLY from files written by the
# earlier modules during this run — no hardcoded values are used.
#
# Usage
# -----
#   python run_pipeline.py                              # canonical SR-1L-POL
#   python run_pipeline.py --levels 1L 2L --opsets SCE SRL POL  # full sweep
#   python run_pipeline.py --skip_train --skip_sr       # finetune + visualize
#   python run_pipeline.py --eval_only                  # charts from existing files
#   python run_pipeline.py --show_plots                 # open charts interactively


import argparse
import pathlib
import sys
import textwrap

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))

import module1_data_model          as M1
import module2_symbolic_regression as M2
import module3_hls_codegen         as M3
import module4_finetune_eval       as M4
import module5_visualization       as M5

# Defaults 
DEFAULT_LEVELS = ["1L"]
DEFAULT_OPSETS = ["POL"]
C_MAX          = 30
HLS4ML_FN      = "myproject"
DIM_MAP        = {"1L": 128, "2L": 512}



# Pipeline


def run_pipeline(levels:      list[str],
                 opsets:      list[str],
                 skip_train:  bool,
                 skip_sr:     bool,
                 skip_hls:    bool,
                 skip_ft:     bool,
                 eval_only:   bool,
                 show_plots:  bool,
                 c_max:       int):

    print("=" * 70)
    print("  NeuSym-HLS Pipeline  -  SVHN Binary Classification (1 vs 7)")
    print("  Target FPGA: AMD XC7Z020 @ 100 MHz")
    print("=" * 70)

    
    # Module 1  -  Data Preparation, MLP Training & Trace Extraction
   
    if not eval_only:
        print("\n" + "-" * 70)
        print("  MODULE 1  -  SVHN Data, MLP Training & Trace Extraction")
        print("-" * 70)
        # run() now returns {"traces": ..., "baseline_accuracy": float}
        # It also writes eval_results/baseline_accuracy.txt for Module 5
        m1_out = M1.run(retrain=not skip_train)
        baseline_acc = m1_out["baseline_accuracy"]
        print(f"  [OK] Baseline accuracy: {baseline_acc:.2f}%")

  
    # Module 2  -  Symbolic Regression
    
    if not eval_only and not skip_sr:
        print("\n" + "-" * 70)
        print("  MODULE 2  -  Hardware-Aware Symbolic Regression (PySR)")
        print("-" * 70)
        # run_sweep writes sr_results/<level>_<opset>/hall_of_fame.csv
        # and best_equation.sympy.pkl for each config
        M2.run_sweep(levels=levels, opsets=opsets, c_max=c_max)

   
    # Module 3  -  HLS C++ Code Generation
   
    if not eval_only and not skip_hls:
        print("\n" + "-" * 70)
        print("  MODULE 3  -  Hybrid Datapath HLS Code Generation")
        print("-" * 70)
        for lvl in levels:
            for ops in opsets:
                sym_pkl = M3.SR_DIR / f"{lvl}_{ops}" / "best_equation.sympy.pkl"
                if not sym_pkl.exists():
                    print(f"  [SKIP] {lvl}_{ops}: no SR output yet (run Module 2 first)")
                    continue
                M3.generate_hls_code(
                    level     = lvl,
                    opset     = ops,
                    dim       = DIM_MAP[lvl],
                    hls4ml_fn = HLS4ML_FN,
                )

    
    # Module 4a  -  Fine-Tuning
  
    if not eval_only and not skip_ft:
        print("\n" + "-" * 70)
        print("  MODULE 4a  -  Hybrid Model Fine-Tuning")
        print("-" * 70)
        for lvl in levels:
            for ops in opsets:
                sym_pkl = M4.SR_DIR / f"{lvl}_{ops}" / "best_equation.sympy.pkl"
                if not sym_pkl.exists():
                    print(f"  [SKIP] {lvl}_{ops}: no SR expression found")
                    continue
                # run_finetune returns (model, acc_pct) and writes
                # eval_results/accuracy_log.csv  <-- Module 5 reads this
                _, acc = M4.run_finetune(level=lvl, opset=ops)
                print(f"  [OK] {lvl}_{ops} fine-tuned accuracy: {acc:.2f}%")

    
    # Module 4b  -  Vitis HLS Synthesis Report (optional)
    
    print("\n" + "-" * 70)
    print("  MODULE 4b  -  Vitis HLS Synthesis Report Parsing")
    print("-" * 70)
    # collect_all_metrics writes eval_results/synthesis_metrics.csv if reports exist
    # If no csynth.xml files are found it prints instructions and continues
    print(M4.VITIS_HLS_FLOW)
    M4.collect_all_metrics()

   
    # Module 5  -  Evaluation Visualization

    # This is always the final step.  It reads from files written by earlier
    # modules in this very run — no hardcoded values are used.
    #   Required (raises error if missing):
    #     eval_results/baseline_accuracy.txt   <- Module 1
    #     eval_results/accuracy_log.csv        <- Module 4 fine-tuning
    #   Optional (chart skipped if missing):
    #     eval_results/synthesis_metrics.csv   <- Module 4 Vitis HLS
    #     sr_results/<tag>/hall_of_fame.csv    <- Module 2
    print("\n" + "-" * 70)
    print("  MODULE 5  -  Evaluation Visualization")
    print("-" * 70)
    M5.run(show=show_plots)

    print("\n" + "=" * 70)
    print("  NeuSym-HLS Pipeline Complete")
    print("  Charts saved to: eval_results/")
    print("=" * 70)



# CLI

def _parse_args():
    p = argparse.ArgumentParser(
        description="NeuSym-HLS end-to-end pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples
        --------
          python run_pipeline.py
              Canonical SR-1L-POL run (train -> SR -> codegen -> finetune -> charts)

          python run_pipeline.py --levels 1L 2L --opsets SCE SRL POL
              Full 6-config sweep

          python run_pipeline.py --skip_train --skip_sr
              Skip training and SR; run HLS codegen, fine-tuning, and charts

          python run_pipeline.py --eval_only
              Only regenerate charts from existing result files

          python run_pipeline.py --show_plots
              Open charts interactively in addition to saving them
        """)
    )
    p.add_argument("--levels",      nargs="+", default=DEFAULT_LEVELS,
                   choices=["1L", "2L"])
    p.add_argument("--opsets",      nargs="+", default=DEFAULT_OPSETS,
                   choices=["SCE", "SRL", "POL"])
    p.add_argument("--c_max",       type=int, default=C_MAX)
    p.add_argument("--skip_train",  action="store_true",
                   help="Load checkpoint instead of retraining")
    p.add_argument("--skip_sr",     action="store_true",
                   help="Skip symbolic regression")
    p.add_argument("--skip_hls",    action="store_true",
                   help="Skip HLS code generation")
    p.add_argument("--skip_ft",     action="store_true",
                   help="Skip fine-tuning")
    p.add_argument("--eval_only",   action="store_true",
                   help="Only regenerate charts from existing result files")
    p.add_argument("--show_plots",  action="store_true",
                   help="Display charts interactively (plt.show)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        levels      = args.levels,
        opsets      = args.opsets,
        skip_train  = args.skip_train,
        skip_sr     = args.skip_sr,
        skip_hls    = args.skip_hls,
        skip_ft     = args.skip_ft,
        eval_only   = args.eval_only,
        show_plots  = args.show_plots,
        c_max       = args.c_max,
    )
