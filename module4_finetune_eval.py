
# module4_finetune_eval.py  –  NeuSym-HLS  |  Module 4
#
# Fine-Tuning & Synthesis Evaluation
#
# Sub-module A  –  Fine-Tuning
# --------------------------------------
#   1. Wrap the sympy symbolic expression in a custom nn.Module (SymbolicLayer)
#      that evaluates the analytic formula using differentiable PyTorch ops.
#   2. Build a HybridMLP: frozen hls4ml-equivalent front-end + SymbolicLayer.
#   3. End-to-end retrain with a small learning rate to close the accuracy gap
#      from the SR distillation step.
#
# Sub-module B  –  Synthesis Evaluation
# ---------------------------------------------------------------------------------
#   1. Parse the Vitis HLS post-synthesis XML report (csynth.xml) to extract:
#        • Estimated clock period (ns) / latency (cycles / µs)
#        • LUT, FF, DSP utilisation (absolute + % of XC7Z020 resources)
#   2. Print a formatted comparison table and save results to eval_results.csv.
#
# XC7Z020 resource limits (Zynq-7000 datasheet)
#------------------------------------------------------
#   LUTs : 53,200   FFs : 106,400   DSPs : 220   BRAMs : 140


import os
import re
import csv
import ast
import math
import pickle
import pathlib
import textwrap
import xml.etree.ElementTree as ET
from typing import Callable, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import sympy

# Paths
ROOT       = pathlib.Path(__file__).parent
SR_DIR     = ROOT / "sr_results"
HLS_DIR    = ROOT / "hls_output"
CKPT_DIR   = ROOT / "checkpoints"
EVAL_DIR   = ROOT / "eval_results"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# FPGA resource limits  (XC7Z020 Zynq-7000)
FPGA_RESOURCES = {
    "LUT":  53_200,
    "FF":  106_400,
    "DSP":     220,
    "BRAM":    140,
}

# Training hyper-parameters for fine-tuning 
FT_LR         = 1e-4     # small LR: front-end is near-optimal already
FT_EPOCHS     = 10
FT_BATCH_SIZE = 64
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Sub-module A  –  SymbolicLayer and fine-tuning


# A1.  Sympy → PyTorch differentiable translator


def _sympy_to_torch_fn(expr: sympy.Expr,
                        dim:  int) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Convert a sympy expression with variables x0, x1, … into a Python callable
    f(h: Tensor[..., dim]) → Tensor[..., 1] that uses only PyTorch ops.

    The returned function is differentiable (all ops are in the autograd graph)
    so it can be used inside an nn.Module for end-to-end gradient training.

    Strategy: walk the sympy AST recursively and build a lambda that calls
    PyTorch primitives.  This avoids the overhead of sympy.lambdify + numpy
    bridge.
    """

    def _convert(node: Any) -> Callable[[torch.Tensor], torch.Tensor]:
        """Recursively convert a sympy node to a Tensor → Tensor lambda."""

        # Leaf: variable symbol xk 
        if isinstance(node, sympy.Symbol):
            name = node.name
            if name.startswith("x_"):
                k = int(name[2:])
            elif name.startswith("x"):
                k = int(name[1:])
            else:
                raise ValueError(f"Unrecognised symbol name: {name!r}")
            if k >= dim:
                raise ValueError(f"Symbol index {k} ≥ input dim {dim}")
            return lambda h, _k=k: h[..., _k : _k + 1]

        # Leaf: numeric constant
        if isinstance(node, (sympy.Integer, sympy.Float,
                              sympy.Rational, sympy.Number)):
            val = float(node)
            return lambda h, _v=val: torch.full(
                (*h.shape[:-1], 1), _v, dtype=h.dtype, device=h.device)

        # Add: sum of child terms
        if isinstance(node, sympy.Add):
            children = [_convert(a) for a in node.args]
            def _add(h, ch=children):
                out = ch[0](h)
                for c in ch[1:]:
                    out = out + c(h)
                return out
            return _add

        # Mul: product of child terms 
        if isinstance(node, sympy.Mul):
            children = [_convert(a) for a in node.args]
            def _mul(h, ch=children):
                out = ch[0](h)
                for c in ch[1:]:
                    out = out * c(h)
                return out
            return _mul

        # Pow: a ** b 
        if isinstance(node, sympy.Pow):
            base_fn = _convert(node.base)
            exp_val = node.exp
            if isinstance(exp_val, sympy.Integer):
                n = int(exp_val)
                if n == 2:
                    return lambda h, b=base_fn: b(h) ** 2
                if n == 3:
                    return lambda h, b=base_fn: b(h) ** 3
                if n == -1:
                    return lambda h, b=base_fn: 1.0 / b(h)
            exp_fn = _convert(exp_val)
            return lambda h, b=base_fn, e=exp_fn: torch.pow(b(h), e(h))

        # sin / cos / exp / log 
        if isinstance(node, sympy.sin):
            ch = _convert(node.args[0])
            return lambda h, c=ch: torch.sin(c(h))

        if isinstance(node, sympy.cos):
            ch = _convert(node.args[0])
            return lambda h, c=ch: torch.cos(c(h))

        if isinstance(node, sympy.exp):
            ch = _convert(node.args[0])
            return lambda h, c=ch: torch.exp(c(h))

        if isinstance(node, sympy.log):
            ch = _convert(node.args[0])
            return lambda h, c=ch: torch.log(c(h).clamp(min=1e-7))

        #  Abs 
        if isinstance(node, sympy.Abs):
            ch = _convert(node.args[0])
            return lambda h, c=ch: torch.abs(c(h))

        # ReLU: Max(0, x) 
        if isinstance(node, sympy.Max):
            children = [_convert(a) for a in node.args]
            def _max(h, ch=children):
                out = ch[0](h)
                for c in ch[1:]:
                    out = torch.maximum(out, c(h))
                return out
            return _max

        # Sqrt 
        if isinstance(node, sympy.sqrt):
            ch = _convert(node.args[0])
            return lambda h, c=ch: torch.sqrt(c(h).clamp(min=0.0))

        raise NotImplementedError(
            f"Cannot convert sympy node type {type(node).__name__}: {node}"
        )

    return _convert(expr)



# A2.  SymbolicLayer  nn.Module


class SymbolicLayer(nn.Module):
    """
    A drop-in nn.Module that evaluates a fixed analytic expression discovered
    by PySR.  The expression is compiled once at construction time into a
    differentiable PyTorch lambda.

    Parameters
    ----------
    sympy_expr : sympy.Expr
        The symbolic expression with variables x0, x1, …, x_{dim-1}.
    dim        : int
        Dimensionality of the input hidden vector.
    learnable_scale : bool
        If True, add a single learnable scalar *s* so the layer outputs
        s * f(h).  This allows fine-tuning to rescale the SR output without
        modifying the symbolic expression structure.
    learnable_bias  : bool
        If True, add a learnable scalar bias b, output = s*f(h) + b.
    """

    def __init__(self,
                 sympy_expr:      sympy.Expr,
                 dim:             int,
                 learnable_scale: bool = True,
                 learnable_bias:  bool = True):
        super().__init__()
        self.dim        = dim
        self.expr_str   = str(sympy_expr)
        self._eval_fn   = _sympy_to_torch_fn(sympy_expr, dim)

        # Optional learnable affine parameters for fine-tuning
        self.scale = nn.Parameter(torch.ones(1))  if learnable_scale else None
        self.bias  = nn.Parameter(torch.zeros(1)) if learnable_bias  else None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h : Tensor shape (B, dim)

        Returns
        -------
        logit : Tensor shape (B, 1)
        """
        out = self._eval_fn(h)                       # (B, 1)
        if self.scale is not None:
            out = self.scale * out
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, "
                f"scale={'learnable' if self.scale is not None else 'fixed'}, "
                f"bias={'learnable' if self.bias is not None else 'fixed'}\n"
                f"expr={self.expr_str[:80]}{'…' if len(self.expr_str)>80 else ''}")



# A3.  HybridMLP  –  frozen front-end + symbolic back-end


class HybridMLP(nn.Module):
    """
    Hybrid model for fine-tuning.

    The front-end layers (fc1, bn1, fc2, bn2) are initially loaded from the
    trained baseline checkpoint and *frozen*.  Only the SymbolicLayer
    parameters (scale, bias) are optimised by default.  To unlock the full
    model, call model.unfreeze_frontend().

    Architecture (1L mode)
    ──────────────────────
        Input (3072) → fc1(ReLU+BN) → fc2(ReLU+BN) → [SymbolicLayer(dim=128)]

    Architecture (2L mode)
    ──────────────────────
        Input (3072) → fc1(ReLU+BN) → [SymbolicLayer(dim=512)]
    """

    def __init__(self,
                 sympy_expr: sympy.Expr,
                 level:      str,              # "1L" or "2L"
                 ckpt_path:  pathlib.Path):
        super().__init__()

        assert level in ("1L", "2L"), f"level must be '1L' or '2L', got {level!r}"
        self.level = level

        # Front-end layers (shared for both levels)
        self.fc1 = nn.Linear(3072, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=False)

        # Symbolic back-end 
        dim = 128 if level == "1L" else 512
        self.sr_layer = SymbolicLayer(sympy_expr, dim=dim)

        # Load baseline checkpoint 
        self._load_baseline(ckpt_path)

        # Freeze front-end initially 
        self.freeze_frontend()

    def _load_baseline(self, ckpt_path: pathlib.Path):
        """
        Load fc1/bn1/fc2/bn2 weights from the baseline MLP checkpoint.
        The final linear (fc3) is discarded — it is replaced by sr_layer.
        """
        state = torch.load(ckpt_path, map_location="cpu")
        # Only load keys that exist in this model (skip fc3.*)
        own_keys = {k for k in self.state_dict() if not k.startswith("sr_layer")}
        filtered = {k: v for k, v in state.items() if k in own_keys}
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        skipped = [k for k in missing if not k.startswith("sr_layer")]
        if skipped:
            print(f"[Module 4] WARNING: Missing keys in checkpoint: {skipped}")
        print(f"[Module 4] Loaded baseline front-end from {ckpt_path}")

    def freeze_frontend(self):
        """Freeze all layers except the SymbolicLayer affine parameters."""
        for name, param in self.named_parameters():
            if not name.startswith("sr_layer"):
                param.requires_grad_(False)
        print("[Module 4] Front-end frozen (only SR scale/bias trainable).")

    def unfreeze_frontend(self):
        """Unfreeze the entire model for full end-to-end fine-tuning."""
        for param in self.parameters():
            param.requires_grad_(True)
        print("[Module 4] All parameters unfrozen for end-to-end fine-tuning.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)                        # flatten

        if self.level == "2L":
            # 2L: replace fc2 + final layer with SR
            x = self.relu(self.bn1(self.fc1(x)))          # (B, 512)
            logit = self.sr_layer(x)                      # (B, 1)
        else:
            # 1L: replace only fc3 with SR
            x = self.relu(self.bn1(self.fc1(x)))          # (B, 512)
            x = self.relu(self.bn2(self.fc2(x)))          # (B, 128)
            logit = self.sr_layer(x)                      # (B, 1)

        return logit



# A4.  Fine-tuning loop

def finetune(model:        nn.Module,
             train_loader: DataLoader,
             test_loader:  DataLoader,
             epochs:       int         = FT_EPOCHS,
             lr:           float       = FT_LR,
             unfreeze_at:  int | None  = None,
             device:       torch.device = DEVICE,
             tag:          str          = "hybrid") -> nn.Module:
    """
    Fine-tune the hybrid model.

    Parameters
    ----------
    model        : HybridMLP (or any nn.Module with a 1-output forward pass)
    train_loader : DataLoader for the 1-vs-7 SVHN training set
    test_loader  : DataLoader for the test set
    epochs       : number of fine-tuning epochs
    lr           : learning rate (should be small, e.g. 1e-4)
    unfreeze_at  : if set, unfreeze the full model at this epoch number
    device       : torch device
    tag          : label for checkpoint and log output

    Returns
    -------
    model : the fine-tuned model (best checkpoint loaded)
    """
    model   = model.to(device)
    criterion = nn.BCEWithLogitsLoss()

    # Only optimise parameters that require grad
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc  = 0.0
    ckpt_path = CKPT_DIR / f"{tag}.pt"

    print(f"\n[Module 4] Fine-tuning on {device} for {epochs} epochs "
          f"(lr={lr}, unfreeze_at={unfreeze_at})\n")
    print(f"{'Epoch':>5} {'Train Loss':>10} {'Train Acc':>10} "
          f"{'Test Loss':>10} {'Test Acc':>10}")
    print("─" * 55)

    for epoch in range(1, epochs + 1):

        #  Optional: unfreeze front-end at a specified epoch
        if unfreeze_at is not None and epoch == unfreeze_at:
            if hasattr(model, "unfreeze_frontend"):
                model.unfreeze_frontend()
            # Rebuild optimizer with all parameters
            optimizer = optim.Adam(
                model.parameters(), lr=lr * 0.1, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - epoch + 1)
            print(f"[Module 4] Epoch {epoch}: front-end unfrozen, lr reset to {lr*0.1}")

        # Train 
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0

        for imgs, targets in train_loader:
            imgs    = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).float().unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss   = criterion(logits, targets)
            loss.backward()
            # Gradient clipping avoids exploding gradients in SR affine params
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            tr_loss    += loss.item() * imgs.size(0)
            preds       = (torch.sigmoid(logits) >= 0.5).long()
            tr_correct += (preds == targets.long()).sum().item()
            tr_total   += imgs.size(0)

        # Evaluate 
        model.eval()
        te_loss, te_correct, te_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, targets in test_loader:
                imgs    = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True).float().unsqueeze(1)
                logits  = model(imgs)
                loss    = criterion(logits, targets)
                te_loss    += loss.item() * imgs.size(0)
                preds       = (torch.sigmoid(logits) >= 0.5).long()
                te_correct += (preds == targets.long()).sum().item()
                te_total   += imgs.size(0)

        tr_acc = tr_correct / tr_total
        te_acc = te_correct / te_total
        scheduler.step()

        print(f"{epoch:>5} {tr_loss/tr_total:>10.4f} {tr_acc:>10.4f} "
              f"{te_loss/te_total:>10.4f} {te_acc:>10.4f}")

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), ckpt_path)

    print(f"\n[Module 4] Best test accuracy after fine-tuning: {best_acc:.4f}")
    print(f"[Module 4] Best checkpoint -> {ckpt_path}\n")

    # Persist accuracy so Module 5 can plot real results 
    # accuracy_log.csv accumulates one row per (level, opset) fine-tune run.
    # If the file already exists we update the row for this tag; otherwise append.
    acc_csv = EVAL_DIR / "accuracy_log.csv"
    acc_pct = best_acc * 100.0

    # Read existing rows (if any), update or insert this tag's row
    existing: list[dict] = []
    if acc_csv.exists():
        with open(acc_csv, newline="") as f:
            existing = list(csv.DictReader(f))

    # Remove stale entry for the same tag so we don't duplicate
    existing = [r for r in existing if r.get("tag") != tag]
    existing.append({"tag": tag, "accuracy": f"{acc_pct:.4f}"})

    with open(acc_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tag", "accuracy"])
        writer.writeheader()
        writer.writerows(existing)

    print(f"[Module 4] Accuracy logged: {tag} = {acc_pct:.2f}%  -> {acc_csv}")

    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model, acc_pct



# Sub-module B  –  Vitis HLS synthesis report parser


# ---------------------------------------------------------------------------
# B1.  XML report parser (csynth.xml produced by Vitis HLS)


def parse_csynth_xml(xml_path: pathlib.Path) -> dict:
    """
    Parse a Vitis HLS post-synthesis report XML file (csynth.xml) and extract
    the key metrics for the NeuSym-HLS evaluation table.

    The csynth.xml schema (Vitis HLS 2022.x–2024.x) places resource data under:
        //AreaEstimates/Resources/{LUT, FF, DSP, BRAM_18K}
    and timing data under:
        //PerformanceEstimates/SummaryOfOverallLatency/{Latency-min, Latency-max}
        //TimingEstimates/EstimatedClockPeriod

    Parameters
    ----------
    xml_path : pathlib.Path to csynth.xml

    Returns
    -------
    metrics : dict with keys
        'latency_min_cycles', 'latency_max_cycles',
        'clock_period_ns', 'latency_min_us', 'latency_max_us',
        'LUT', 'FF', 'DSP', 'BRAM',
        'LUT_pct', 'FF_pct', 'DSP_pct', 'BRAM_pct'
    """
    if not xml_path.exists():
        raise FileNotFoundError(f"csynth.xml not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    def _find_text(path: str, default: str = "0") -> str:
        node = root.find(path)
        return node.text.strip() if node is not None and node.text else default

    # Timing 
    clock_ns = float(_find_text(
        ".//TimingEstimates/EstimatedClockPeriod", str(CLOCK_PERIOD)))
    lat_min  = int(_find_text(
        ".//PerformanceEstimates/SummaryOfOverallLatency/Latency-min", "0"))
    lat_max  = int(_find_text(
        ".//PerformanceEstimates/SummaryOfOverallLatency/Latency-max", "0"))

    # Resources 
    # Vitis HLS 2022+ uses AreaEstimates/Resources; fall back to older paths
    base = ".//AreaEstimates/Resources"
    lut   = int(_find_text(f"{base}/LUT",     "0"))
    ff    = int(_find_text(f"{base}/FF",      "0"))
    dsp   = int(_find_text(f"{base}/DSP",     "0"))
    bram  = int(_find_text(f"{base}/BRAM_18K","0"))

    metrics = {
        "latency_min_cycles": lat_min,
        "latency_max_cycles": lat_max,
        "clock_period_ns":    clock_ns,
        "latency_min_us":     lat_min * clock_ns / 1000.0,
        "latency_max_us":     lat_max * clock_ns / 1000.0,
        "LUT":    lut,
        "FF":     ff,
        "DSP":    dsp,
        "BRAM":   bram,
        "LUT_pct":  100.0 * lut  / FPGA_RESOURCES["LUT"],
        "FF_pct":   100.0 * ff   / FPGA_RESOURCES["FF"],
        "DSP_pct":  100.0 * dsp  / FPGA_RESOURCES["DSP"],
        "BRAM_pct": 100.0 * bram / FPGA_RESOURCES["BRAM"],
    }
    return metrics



# B2.  Pretty-print and CSV export


def print_metrics(tag: str, metrics: dict):
    """Print a formatted synthesis metrics table."""
    print(f"\n╔══ Synthesis Report: {tag} ══{'═'*max(0,40-len(tag))}╗")
    print(f"║  Clock period  : {metrics['clock_period_ns']:.2f} ns  "
          f"({1000/metrics['clock_period_ns']:.0f} MHz)")
    print(f"║  Latency (min) : {metrics['latency_min_cycles']} cycles  "
          f"({metrics['latency_min_us']:.3f} µs)")
    print(f"║  Latency (max) : {metrics['latency_max_cycles']} cycles  "
          f"({metrics['latency_max_us']:.3f} µs)")
    print(f"╠{'═'*55}╣")
    print(f"║  {'Resource':<8} {'Used':>8} {'Total':>8} {'%':>8}")
    print(f"║  {'─'*36}")
    for res in ("LUT", "FF", "DSP", "BRAM"):
        used  = metrics[res]
        total = FPGA_RESOURCES[res]
        pct   = metrics[f"{res}_pct"]
        bar   = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"║  {res:<8} {used:>8,} {total:>8,} {pct:>7.1f}%  {bar}")
    print(f"╚{'═'*55}╝\n")


def collect_all_metrics(hls_dir: pathlib.Path = HLS_DIR) -> list[dict]:
    """
    Walk hls_output/ to find all csynth.xml reports and aggregate metrics.

    Expected path pattern:
      hls_output/<level>_<opset>/hybrid_neusym_<level>/solution1/syn/report/csynth.xml
    """
    rows = []
    for run_dir in sorted(hls_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        # glob for csynth.xml anywhere under this run directory
        for xml_path in run_dir.rglob("csynth.xml"):
            tag = run_dir.name
            try:
                m = parse_csynth_xml(xml_path)
                m["tag"] = tag
                print_metrics(tag, m)
                rows.append(m)
            except Exception as exc:
                print(f"[Module 4] Could not parse {xml_path}: {exc}")

    if not rows:
        print("[Module 4] No csynth.xml files found — run Vitis HLS synthesis first.\n"
              "           Command:  cd hls_output/<level>_<opset>/tcl && "
              "vitis_hls -f synth.tcl")
        return rows

    # Save CSV 
    csv_path = EVAL_DIR / "synthesis_metrics.csv"
    fieldnames = ["tag",
                  "latency_min_cycles", "latency_max_cycles",
                  "clock_period_ns", "latency_min_us", "latency_max_us",
                  "LUT", "LUT_pct", "FF", "FF_pct",
                  "DSP", "DSP_pct", "BRAM", "BRAM_pct"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Module 4] Metrics CSV → {csv_path}")
    return rows



# B3.  Vitis HLS flow summary (reference)


VITIS_HLS_FLOW = textwrap.dedent("""
╔══════════════════════════════════════════════════════════════════════════════╗
║          NeuSym-HLS  –  Vitis HLS Synthesis Flow Reference                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Step 0: Prerequisites                                                       ║
║  ─────────────────────                                                       ║
║  • Vitis HLS 2022.1+ installed and on PATH                                  ║
║  • hls4ml installed: pip install hls4ml                                      ║
║  • Module 1–3 outputs present                                                ║
║                                                                              ║
║  Step 1: Generate hls4ml front-end                                           ║
║  ─────────────────────────────────                                           ║
║    python -c "                                                               ║
║      import hls4ml, torch                                                   ║
║      from module1_data_model import MLP_SVHN                                ║
║      model = MLP_SVHN()                                                      ║
║      model.load_state_dict(torch.load('checkpoints/mlp_svhn.pt'))           ║
║      # Strip fc3 for 1L; strip fc2+fc3 for 2L                               ║
║      cfg = hls4ml.utils.config_from_pytorch_model(model, granularity='name')║
║      hls_model = hls4ml.converters.convert_from_pytorch_model(              ║
║                      model, hls_config=cfg, output_dir='hls4ml_prj',        ║
║                      part='xc7z020clg400-1', clock_period=10)               ║
║      hls_model.compile()    # generates hls4ml_prj/firmware/                ║
║    "                                                                         ║
║                                                                              ║
║  Step 2: Generate NeuSym symbolic layer (Module 3)                           ║
║  ──────────────────────────────────────────────────                          ║
║    python module3_hls_codegen.py --level 1L --opset POL                     ║
║                                                                              ║
║  Step 3: Run Vitis HLS synthesis                                              ║
║  ─────────────────────────────────                                           ║
║    cd hls_output/1L_POL/tcl                                                 ║
║    vitis_hls -f synth.tcl                                                   ║
║                                                                              ║
║  Step 4: Parse synthesis report (Module 4)                                   ║
║  ─────────────────────────────────────────                                   ║
║    python module4_finetune_eval.py --eval_only                              ║
║                                                                              ║
║  Report location after csynth_design:                                        ║
║    hls_output/1L_POL/hybrid_neusym_1L/solution1/syn/report/csynth.xml      ║
║                                                                              ║
║  Vivado implementation (for placed-and-routed resource numbers):             ║
║    open_project hybrid_neusym_1L                                            ║
║    open_solution solution1                                                  ║
║    export_design -format ip_catalog                                          ║
║    # Then import IP into Vivado project targeting xc7z020clg400-1           ║
║    # Run Synthesis + Implementation → check Utilization report               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")



# Entry point


def run_finetune(level: str = "1L", opset: str = "POL"):
    """Load SR expression + baseline model and run fine-tuning."""
    # Load data 
    # Import here to avoid circular imports in standalone usage
    from module1_data_model import build_dataloaders
    train_loader, test_loader = build_dataloaders(batch_size=FT_BATCH_SIZE)

    # Load sympy expression 
    sym_path = SR_DIR / f"{level}_{opset}" / "best_equation.sympy.pkl"
    if not sym_path.exists():
        raise FileNotFoundError(
            f"Sympy pickle not found: {sym_path}\n"
            "Run Module 2 first.")
    with open(sym_path, "rb") as f:
        sympy_expr = pickle.load(f)

    # Build hybrid model 
    ckpt_path = CKPT_DIR / "mlp_svhn.pt"
    model = HybridMLP(sympy_expr=sympy_expr, level=level, ckpt_path=ckpt_path)

    # Fine-tune (phase 1: SR affine params only) 
    # finetune() now returns (model, best_accuracy_pct) and writes accuracy_log.csv
    model, acc_pct = finetune(
        model, train_loader, test_loader,
        epochs      = FT_EPOCHS,
        lr          = FT_LR,
        unfreeze_at = 6,         # unfreeze front-end at epoch 6 for joint tuning
        tag         = f"{level}_{opset}",   # tag matches level_opset key used by Module 5
    )
    return model, acc_pct


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="NeuSym-HLS Module 4 – Fine-Tuning & Evaluation")
    parser.add_argument("--level",     choices=["1L", "2L"], default="1L")
    parser.add_argument("--opset",     choices=["SCE", "SRL", "POL"], default="POL")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip fine-tuning; only parse HLS synthesis reports")
    args = parser.parse_args()

    print(VITIS_HLS_FLOW)

    if args.eval_only:
        collect_all_metrics()
    else:
        run_finetune(level=args.level, opset=args.opset)
        collect_all_metrics()
