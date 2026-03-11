# NeuSym-HLS: Neurosymbolic Inference Accelerator for SVHN

A complete Python and C++ pipeline that replaces computationally expensive terminal layers of a trained deep neural network with compact analytic expressions discovered via symbolic regression, producing a hybrid neurosymbolic FPGA accelerator with significantly reduced resource utilisation and near-baseline accuracy.

**Target Hardware:** AMD XC7Z020 FPGA (Zynq-7000) @ 100 MHz
**Dataset:** SVHN (Street View House Numbers) — Binary classification: digit 1 vs digit 7
**Key Result (SR-1L-POL):** 98.2% accuracy using only 5% LUTs and 7% DSPs

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Running the Pipeline](#running-the-pipeline)
- [Module Descriptions](#module-descriptions)
- [Output Files](#output-files)
- [Results](#results)
- [University Server / HPC](#university-server--hpc)
- [Requirements](#requirements)

---

## Overview

Standard neural network inference on FPGAs is expensive — a 3072→512→128→1 MLP requires hundreds of multiply-accumulate operations mapped to DSP slices. **NeuSym-HLS** addresses this by:

1. Training a full FP32 baseline MLP on SVHN
2. Extracting the hidden activations flowing into the final layer(s)
3. Running **hardware-aware symbolic regression** (PySR) to find a compact analytic formula that maps those activations directly to the output logit
4. Auto-generating **synthesisable Vitis HLS C++** from the discovered formula
5. Stitching the symbolic C++ layer to a quantised hls4ml front-end to form a complete hybrid accelerator
6. Fine-tuning the hybrid model end-to-end to recover any accuracy lost during distillation
7. Visualising the accuracy vs. hardware area trade-off across all configurations

The genetic search in PySR is biased toward hardware-efficient operators by assigning each operator a **clock-cycle cost** derived from Xilinx floating-point IP datasheets, steering the Pareto front toward expressions that map cheaply to LUTs and DSP slices.

---

## Project Structure

```
neusym_hls/
│
├── run_pipeline.py                  # Top-level orchestrator (run this)
│
├── module1_data_model.py            # SVHN loading, MLP training, trace extraction
├── module2_symbolic_regression.py   # Hardware-aware PySR symbolic regression
├── module3_hls_codegen.py           # Sympy → Vitis HLS C++ code generator
├── module4_finetune_eval.py         # Fine-tuning + synthesis report parser
├── module5_visualization.py         # Evaluation charts (reads live pipeline results)
│
├── utils/
│   ├── __init__.py
│   └── hw_costs.py                  # FPGA operator latency cost table
│
├── .gitignore
└── README.md
```

**Generated at runtime (not tracked by git):**
```
data/                    # SVHN dataset (auto-downloaded)
checkpoints/             # Trained model weights (.pt)
traces/                  # Hidden activation arrays (.npy)
sr_results/              # PySR hall of fame, sympy expressions
hls_output/              # Generated C++, headers, TCL synthesis scripts
eval_results/            # Charts (.png, .pdf), metrics CSVs
```

---

## How It Works

### Architecture

```
SVHN Image (32x32x3)
        |
        v
  [fc1: 3072->512, ReLU+BN]         <- always runs as quantised MLP (hls4ml)
        |
        |-- 2L distillation point (h ∈ R^512)
        v
  [fc2: 512->128,  ReLU+BN]         <- kept in 1L mode, replaced in 2L mode
        |
        |-- 1L distillation point (h ∈ R^128)
        v
  [Symbolic SR Layer: f(h) -> R^1]   <- replaces fc3 (or fc2+fc3)
        |
        v
   Binary logit (1 vs 7)
```

### Distillation Levels

| Level | Layers replaced | Hidden dim | Complexity |
|-------|----------------|------------|------------|
| **1L** | fc3 only | 128 | Lower — replaces 1 layer |
| **2L** | fc2 + fc3 | 512 | Higher — replaces 2 layers |

### Operator Sets

| Set | Operators | FPGA cost | Best use case |
|-----|-----------|-----------|---------------|
| **POL** | +, -, *, / | Lowest (DSP chains only) | Maximum resource savings |
| **SRL** | +, -, *, square, relu | Low (no CORDIC) | Good accuracy/area balance |
| **SCE** | +, -, *, sin, cos, exp | High (CORDIC units) | Richest expressions |

### Hardware Cost Biasing

Each operator is assigned a latency cost in clock cycles:

| Operator | Cost | Reason |
|----------|------|--------|
| +, - | 1 | Simple adder |
| * | 2 | Single DSP slice |
| square | 2 | One multiplier |
| relu | 1 | Comparator only |
| sin, cos | 8 | 8-stage CORDIC pipeline |
| exp | 12 | Range reduction + polynomial |
| / | 16 | Iterative Newton-Raphson |

PySR uses these costs in its `complexity_of_operators` parameter, so the genetic search naturally favours expressions that are cheap to implement in silicon.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/swarai-code/NeuSym-HLS-for-SVHN.git
cd NeuSym-HLS-for-SVHN
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install Python dependencies

```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pysr==0.19.* sympy numpy matplotlib
```

### 4. Install Julia (required by PySR)

Download from [julialang.org](https://julialang.org/downloads/) and add to PATH, then:

```bash
python3 -c "import pysr; pysr.install()"
```

> This only needs to be done once. It installs the SymbolicRegression.jl Julia package.

### 5. (Optional) Vitis HLS — for FPGA synthesis

Required only for Module 4b (hardware resource extraction). Download from [Xilinx/AMD](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis.html). Free for non-commercial use.

---

## Running the Pipeline

### Canonical run — SR-1L-POL (paper's best result)

```bash
python run_pipeline.py
```

This runs all five modules in sequence and saves evaluation charts to `eval_results/`.

### Full sweep — all 6 (level × opset) configurations

```bash
python run_pipeline.py --levels 1L 2L --opsets SCE SRL POL
```

### Skip training if checkpoint already exists

```bash
python run_pipeline.py --skip_train
```

### Skip SR if PySR results already exist

```bash
python run_pipeline.py --skip_train --skip_sr
```

### Regenerate charts only (from existing result files)

```bash
python run_pipeline.py --eval_only
```

### Open charts interactively after saving

```bash
python run_pipeline.py --show_plots
```

### All CLI flags

| Flag | Description |
|------|-------------|
| `--levels 1L 2L` | Distillation levels to process |
| `--opsets SCE SRL POL` | Operator sets to sweep |
| `--c_max N` | Hardware complexity budget (default: 30) |
| `--skip_train` | Load existing checkpoint instead of retraining |
| `--skip_sr` | Skip symbolic regression |
| `--skip_hls` | Skip HLS code generation |
| `--skip_ft` | Skip fine-tuning |
| `--eval_only` | Only regenerate charts from existing files |
| `--show_plots` | Display charts interactively |

---

## Module Descriptions

### Module 1 — `module1_data_model.py`
- Downloads SVHN via torchvision and filters to digits 1 and 7
- Trains a 3072→512→128→1 MLP with BCEWithLogitsLoss and Adam (lr=1e-3)
- Registers forward hooks on `fc3` (1L) and `fc2` (2L) to capture hidden activations
- Saves trace arrays to `traces/` and baseline accuracy to `eval_results/baseline_accuracy.txt`

### Module 2 — `module2_symbolic_regression.py`
- Runs PySR (niterations=40, populations=10) on the extracted traces
- Sweeps across operator sets SCE / SRL / POL with hardware clock-cycle cost biasing
- Applies a complexity budget C_max=30 to filter the Pareto front
- Saves hall of fame CSV and best sympy expression pickle per configuration

### Module 3 — `module3_hls_codegen.py`
- Walks the sympy AST and generates synthesisable Vitis HLS C++ (float datatype)
- Emits `#pragma HLS INLINE` to fold the SR layer into the parent pipeline
- Generates `hybrid_top.cpp` which stitches the hls4ml quantised front-end to the SR layer via AXI-Stream interfaces
- Writes a complete `tcl/synth.tcl` script for Vitis HLS project creation and synthesis

### Module 4 — `module4_finetune_eval.py`
- Converts the sympy expression to differentiable PyTorch ops using a recursive AST walker
- Wraps it in a `SymbolicLayer` nn.Module with learnable scale and bias parameters
- Builds `HybridMLP` by loading the baseline front-end weights and substituting the SR layer
- Fine-tunes in two phases: SR affine params only (epochs 1-5), then full model (epochs 6-10)
- Saves fine-tuned accuracy to `eval_results/accuracy_log.csv`
- Parses Vitis HLS `csynth.xml` reports to extract LUT, FF, DSP, latency metrics

### Module 5 — `module5_visualization.py`
Reads exclusively from files written by earlier modules and generates four charts:

| Chart | Source data | Generated when |
|-------|-------------|----------------|
| `area_vs_accuracy.png` | accuracy_log.csv + synthesis_metrics.csv | Always |
| `resource_breakdown.png` | synthesis_metrics.csv | After Vitis HLS synthesis |
| `pareto_scatter.png` | accuracy_log.csv + synthesis_metrics.csv | Always |
| `sr_complexity.png` | hall_of_fame.csv | After Module 2 |

---

## Output Files

After a full pipeline run:

```
eval_results/
├── baseline_accuracy.txt       # FP32 MLP test accuracy
├── accuracy_log.csv            # Fine-tuned accuracy per (level, opset)
├── synthesis_metrics.csv       # LUT/FF/DSP/latency from Vitis HLS
├── area_vs_accuracy.png/.pdf   # Main trade-off chart
├── resource_breakdown.png/.pdf # Per-resource utilisation breakdown
├── pareto_scatter.png/.pdf     # Accuracy vs area Pareto scatter
└── sr_complexity.png/.pdf      # SR complexity and MSE per config

hls_output/1L_POL/
├── sr_layer.h / sr_layer.cpp   # Synthesisable symbolic function
├── hybrid_top.h / hybrid_top.cpp  # Full accelerator top-level
├── tb_hybrid_top.cpp           # C-simulation testbench
└── tcl/synth.tcl               # Vitis HLS synthesis script
```

---

## Results

| Configuration | Accuracy | LUT | DSP | Latency |
|--------------|----------|-----|-----|---------|
| FP32 Baseline | 98.9% | 100% | 100% | — |
| **SR-1L-POL** | **98.2%** | **5%** | **7%** | **< 1 µs** |
| SR-1L-SRL | 97.9% | 9% | — | — |
| SR-1L-SCE | 97.6% | 18% | — | — |
| SR-2L-POL | 97.4% | 11% | — | — |
| SR-2L-SRL | 97.1% | 17% | — | — |
| SR-2L-SCE | 96.8% | 31% | — | — |

> Hardware numbers are post-synthesis estimates on AMD XC7Z020 (53,200 LUTs, 220 DSPs) @ 100 MHz.

---

## University Server / HPC

### Connect and set up

```bash
ssh <netid>@<server-address>
git clone https://github.com/swarai-code/NeuSym-HLS-for-SVHN.git
cd NeuSym-HLS-for-SVHN
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision pysr==0.19.* sympy numpy matplotlib
python3 -c "import pysr; pysr.install()"
```

### Submit as a SLURM job

```bash
#!/bin/bash
#SBATCH --job-name=neusym_hls
#SBATCH --output=logs/neusym_%j.out
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

source ~/NeuSym-HLS-for-SVHN/venv/bin/activate
cd ~/NeuSym-HLS-for-SVHN
python run_pipeline.py --levels 1L 2L --opsets SCE SRL POL
```

```bash
mkdir -p logs
sbatch run_job.sh
```

### Copy results back to your laptop

```bash
scp -r <netid>@<server>:~/NeuSym-HLS-for-SVHN/eval_results/ ./
```

---

## Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.10+ | Runtime |
| PyTorch | 2.0+ | MLP training and fine-tuning |
| torchvision | 0.15+ | SVHN dataset download |
| PySR | 0.19.* | Symbolic regression (Julia backend) |
| Julia | 1.9+ | PySR backend runtime |
| sympy | 1.12+ | Expression manipulation and code generation |
| numpy | 1.24+ | Array operations |
| matplotlib | 3.7+ | Evaluation charts |
| hls4ml | 0.8+ | Optional — quantised MLP front-end synthesis |
| Vitis HLS | 2022.1+ | Optional — FPGA synthesis and resource extraction |

---

## Citation

If you use this framework in your work, please cite the original NeuSym-HLS paper and this implementation:

```
NeuSym-HLS: Neurosymbolic Inference Acceleration via Hardware-Aware
Symbolic Regression and High-Level Synthesis
Implementation for SVHN Binary Classification on AMD XC7Z020
```
