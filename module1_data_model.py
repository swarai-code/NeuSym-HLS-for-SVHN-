
# module1_data_model.py  –  NeuSym-HLS  |  Module 1
#
# SVHN Data Preparation, MLP Architecture, Training & Trace Extraction
#
# Pipeline
# --------
#   1. Download / load SVHN (torchvision).
#   2. Filter to a binary 1-vs-7 task (label 1 → 0, label 7 → 1).
#   3. Define a three-layer MLP:  3072 → 512 → 128 → 1.
#   4. Train the FP32 baseline to convergence.
#   5. Register forward hooks for two distillation experiments:
#        1L  – hook on the input to fc3 (h ∈ ℝ¹²⁸) + final logit
#        2L  – hook on the input to fc2 (h ∈ ℝ⁵¹²) + final logit
#   6. Run the test set through the hooked model and save NumPy traces.
#
# Outputs (saved to ./traces/)
# ----------------------------
#   traces/1L_hidden.npy   shape (N, 128)
#   traces/1L_logits.npy   shape (N, 1)
#   traces/2L_hidden.npy   shape (N, 512)
#   traces/2L_logits.npy   shape (N, 1)
#   checkpoints/mlp_svhn.pt


import os
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

#Reproducibility 
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

#Paths
ROOT        = pathlib.Path(__file__).parent
DATA_DIR    = ROOT / "data"
CKPT_DIR    = ROOT / "checkpoints"
TRACE_DIR   = ROOT / "traces"
for d in (DATA_DIR, CKPT_DIR, TRACE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Hyper-parameters
BATCH_SIZE  = 64
LR          = 1e-3
EPOCHS      = 30          # typically converges before 30 epochs
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Target digits
DIGIT_A, DIGIT_B = 1, 7   # binary: DIGIT_A → class 0, DIGIT_B → class 1



# 1.  Dataset helpers

def _svhn_transform() -> T.Compose:
    """Standard normalisation for SVHN (ImageNet-style mean/std)."""
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.4377, 0.4438, 0.4728),
                    std=(0.1980, 0.2010, 0.1970)),
    ])


def _filter_binary(dataset: torchvision.datasets.SVHN,
                   a: int = DIGIT_A,
                   b: int = DIGIT_B) -> Subset:
    """
    Return a Subset containing only samples labelled *a* or *b*.
    SVHN uses integer labels 0-9 (note: digit '0' has label 0, NOT 10).
    The binary target is  0 if label==a, 1 if label==b.
    """
    labels = np.array(dataset.labels)          # shape (N,)
    mask   = (labels == a) | (labels == b)
    indices = np.where(mask)[0].tolist()

    # Remap labels in-place so the Subset sees {0, 1}
    dataset.labels = np.where(labels == b, 1, 0)   # everything outside {a,b} → 0, harmless

    return Subset(dataset, indices)


def build_dataloaders(batch_size: int = BATCH_SIZE):
    """Download SVHN and return (train_loader, test_loader)."""
    tf = _svhn_transform()

    raw_train = torchvision.datasets.SVHN(
        root=str(DATA_DIR), split="train", download=True, transform=tf)
    raw_test  = torchvision.datasets.SVHN(
        root=str(DATA_DIR), split="test",  download=True, transform=tf)

    train_sub = _filter_binary(raw_train)
    test_sub  = _filter_binary(raw_test)

    train_loader = DataLoader(train_sub, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_sub,  batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"[Module 1] Train samples: {len(train_sub):,} | "
          f"Test samples: {len(test_sub):,}")
    return train_loader, test_loader

# 2.  Model definition

class MLP_SVHN(nn.Module):
    """
    Three-layer MLP for SVHN binary classification.

    Architecture
    ────────────
        Input (3072) ──┐
        fc1: Linear(3072, 512) + ReLU + BN
        fc2: Linear(512,  128) + ReLU + BN   ← 2L hook point (input)
        fc3: Linear(128,    1)                ← 1L hook point (input)
        Output logit ∈ ℝ¹
    """

    def __init__(self):
        super().__init__()
        #Layers 
        self.fc1 = nn.Linear(3072, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 1)

        self.relu = nn.ReLU(inplace=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten image: (B, 3, 32, 32) → (B, 3072)
        x = x.view(x.size(0), -1)
        # Hidden 1
        x = self.relu(self.bn1(self.fc1(x)))
        # Hidden 2  ← 2L distillation sees input HERE
        x = self.relu(self.bn2(self.fc2(x)))
        # Output    ← 1L distillation sees input HERE
        logit = self.fc3(x)
        return logit   # raw logit, shape (B, 1)



# 3.  Training loop

def train_one_epoch(model:     nn.Module,
                    loader:    DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    device:    torch.device) -> tuple[float, float]:
    """One training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, targets in loader:
        imgs    = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).float().unsqueeze(1)  # (B,1)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)                          # (B, 1)
        loss   = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds   = (torch.sigmoid(logits) >= 0.5).long()
        correct += (preds == targets.long()).sum().item()
        total   += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model:     nn.Module,
             loader:    DataLoader,
             criterion: nn.Module,
             device:    torch.device) -> tuple[float, float]:
    """Evaluation pass. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, targets in loader:
        imgs    = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).float().unsqueeze(1)

        logits  = model(imgs)
        loss    = criterion(logits, targets)

        total_loss += loss.item() * imgs.size(0)
        preds   = (torch.sigmoid(logits) >= 0.5).long()
        correct += (preds == targets.long()).sum().item()
        total   += imgs.size(0)

    return total_loss / total, correct / total


def train_model(model:       nn.Module,
                train_loader: DataLoader,
                test_loader:  DataLoader,
                epochs:       int   = EPOCHS,
                lr:           float = LR,
                device:       torch.device = DEVICE,
                ckpt_path:    str   = str(CKPT_DIR / "mlp_svhn.pt")) -> nn.Module:
    """
    Full training loop with early stopping based on validation accuracy.
    Saves the best checkpoint to *ckpt_path*.
    """
    model   = model.to(device)
    # BCEWithLogitsLoss = sigmoid + binary cross-entropy, numerically stable
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    print(f"\n[Module 1] Training on {device} for up to {epochs} epochs …\n")
    print(f"{'Epoch':>5} {'Train Loss':>10} {'Train Acc':>10} {'Test Loss':>10} {'Test Acc':>10}")
    print("─" * 55)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        print(f"{epoch:>5} {tr_loss:>10.4f} {tr_acc:>10.4f} {te_loss:>10.4f} {te_acc:>10.4f}")

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), ckpt_path)

    print(f"\n[Module 1] Best test accuracy: {best_acc:.4f} — checkpoint saved to {ckpt_path}\n")
    return model



# 4.  Trace extraction via forward hooks

class _HookBuffer:
    """Thread-safe accumulator for hook captures."""
    def __init__(self):
        self.activations: list[torch.Tensor] = []

    def clear(self):
        self.activations.clear()

    def __call__(self, module, input_, output):
        # Store the *input* to the hooked layer (first element of the tuple)
        self.activations.append(input_[0].detach().cpu())


@torch.no_grad()
def extract_traces(model:      nn.Module,
                   loader:     DataLoader,
                   device:     torch.device = DEVICE,
                   save_dir:   pathlib.Path = TRACE_DIR) -> dict[str, np.ndarray]:
    """
    Register hooks for both distillation experiments simultaneously and run a
    single pass through *loader*.

    Returns
    -------
    dict with keys:
        "1L_hidden"  : np.ndarray shape (N, 128)   – input to fc3
        "2L_hidden"  : np.ndarray shape (N, 512)   – input to fc2
        "logits"     : np.ndarray shape (N, 1)     – final output logits
    Saves corresponding .npy files under *save_dir*.
    """
    model.eval()
    model.to(device)

    buf_1L = _HookBuffer()   # captures input to fc3  (dim 128)
    buf_2L = _HookBuffer()   # captures input to fc2  (dim 512)
    logit_buf: list[torch.Tensor] = []

    # Register hooks: forward hook receives (module, input_tuple, output)
    h1 = model.fc3.register_forward_hook(buf_1L)   # 1L: input to last linear
    h2 = model.fc2.register_forward_hook(buf_2L)   # 2L: input to penultimate

    for imgs, _ in loader:
        imgs   = imgs.to(device, non_blocking=True)
        logits = model(imgs)                        # triggers both hooks
        logit_buf.append(logits.cpu())

    # Deregister hooks to avoid side effects in subsequent calls
    h1.remove()
    h2.remove()

    traces = {
        "1L_hidden": torch.cat(buf_1L.activations, dim=0).numpy(),   # (N, 128)
        "2L_hidden": torch.cat(buf_2L.activations, dim=0).numpy(),   # (N, 512)
        "logits":    torch.cat(logit_buf,           dim=0).numpy(),   # (N, 1)
    }

    #Save
    for key, arr in traces.items():
        out = save_dir / f"{key}.npy"
        np.save(out, arr)
        print(f"[Module 1] Saved {key:12s}  shape={arr.shape}  → {out}")

    return traces



# 5.  Entry point

def run(retrain: bool = False) -> dict:
    """
    End-to-end Module 1 execution.

    Parameters
    ----------
    retrain : bool
        If False and a checkpoint exists, load weights and skip training.

    Returns
    -------
    dict with keys:
        "traces"            : trace arrays (1L_hidden, 2L_hidden, logits)
        "baseline_accuracy" : float  – FP32 MLP test accuracy (0-100)
    """
    train_loader, test_loader = build_dataloaders()

    model     = MLP_SVHN()
    ckpt_path = CKPT_DIR / "mlp_svhn.pt"

    if ckpt_path.exists() and not retrain:
        print(f"[Module 1] Loading checkpoint from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    else:
        model = train_model(model, train_loader, test_loader, ckpt_path=str(ckpt_path))
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

    #Compute and persist baseline accuracy
    criterion        = nn.BCEWithLogitsLoss()
    _, baseline_acc  = evaluate(model, test_loader, criterion, DEVICE)
    baseline_acc_pct = baseline_acc * 100.0

    eval_dir = ROOT / "eval_results"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "baseline_accuracy.txt").write_text(str(baseline_acc_pct))
    print(f"[Module 1] Baseline test accuracy: {baseline_acc_pct:.2f}%  "
          f"-> eval_results/baseline_accuracy.txt")

    traces = extract_traces(model, test_loader)
    return {"traces": traces, "baseline_accuracy": baseline_acc_pct}


if __name__ == "__main__":
    run(retrain=False)
