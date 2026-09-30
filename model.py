"""Model 5 — 5-layer Deep Belief Network for learned feature extraction.

Extends the original scaffold (3-layer DBN, randomly initialised RBM stack
with no training loop) to a **5-layer stack** with greedy CD-1 pretraining
and supervised fine-tuning of the regression head:

    RBM(64→48) → RBM(48→32) → RBM(32→16) → RBM(16→8) → Linear(8→1) [RUL]

Pipeline
--------
1. Synthesize a fleet campaign: 48 turbines with turbine-specific wear
   exponents and service lives, inspected every 6 campaign days. Each
   inspection yields 64 sensor channels (wear state, wear rate, load and
   thermal oscillations, interactions, noise) labelled with normalized
   remaining useful life.
2. Split inspections **chronologically by campaign day** (65 / 15 / 20) —
   training never sees an inspection recorded after its cutoff.
3. Greedy layer-wise CD-1 pretraining of the four RBMs (the training loop
   the original scaffold omitted): visible units see thresholded sensor
   bits and hidden samples are chained upward, as in classic DBNs.
4. Fine-tune the stacked network + regression head on mean-field
   probabilities with MSE loss (differential learning rates, early
   stopping on the validation window).
5. Report RMSE / MAE / R² on train/val/test in normalized RUL units and in
   campaign days; persist metrics + best weights.

Run:
    python model.py

The wt-pm platform imports this file through adapter ``m09-dbn-features``
and uses ``RBM``/``DBN`` with the original signatures (``DBN(layers=[...])``,
``rbm.W``/``rbm.v_bias``/``rbm.h_bias``, ``sample_h``/``sample_v``) — keep
those names stable. ``DBN.pretrain``/``DBN.encode`` are additive helpers;
the platform supplies its own CD-1 loop and is unaffected by them.
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SEED = 42
N_TURBINES = 48            # synthetic fleet size
INSPECTION_DAYS = 6        # inspection cadence along each turbine's life
LIFE_RANGE = (180.0, 360.0)   # service life in campaign days
WEAR_EXPONENT_RANGE = (1.2, 2.0)  # turbine-specific wear exponent p
START_RANGE = (0.0, 120.0)    # campaign day each turbine enters service
MEAN_LIFE_DAYS = 270.0     # (180 + 360) / 2 — days scale for reported errors
VAL_FRACTION = 0.15
TEST_FRACTION = 0.20
PRETRAIN_EPOCHS = 15       # CD-1 epochs per RBM
PRETRAIN_LR = 0.1
PRETRAIN_BATCH = 64
FINE_TUNE_EPOCHS = 200
STACK_LR = 1e-3            # fine-tune: RBM stack
HEAD_LR = 1e-2             # fine-tune: regression head
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 20
BATCH_SIZE = 64
RESULTS_DIR = "results"
WEIGHTS_NAME = "model5_weights.pt"
METRICS_NAME = "model5_metrics.json"

torch.manual_seed(SEED)
np.random.seed(SEED)


# ----------------------------------------------------------------------------
# Model 5 architecture
# ----------------------------------------------------------------------------
class RBM(nn.Module):
    """Bernoulli-Bernoulli Restricted Boltzmann Machine.

    Public surface (used by the wt-pm platform adapter ``m09-dbn-features``):
    ``W``, ``v_bias``, ``h_bias`` parameters plus ``sample_h`` / ``sample_v``.
    """

    def __init__(self, visible, hidden):
        super().__init__()
        self.W = nn.Parameter(torch.randn(visible, hidden) * 0.1)
        self.v_bias = nn.Parameter(torch.zeros(visible))
        self.h_bias = nn.Parameter(torch.zeros(hidden))

    def sample_h(self, v):
        prob = torch.sigmoid(v @ self.W + self.h_bias)
        return prob, torch.bernoulli(prob)

    def sample_v(self, h):
        prob = torch.sigmoid(h @ self.W.t() + self.v_bias)
        return prob, torch.bernoulli(prob)

    def free_energy(self, v):
        """F(v) = -v·c - Σ softplus(vW + b); used by the CD-1 update."""
        return -(v @ self.v_bias) - torch.nn.functional.softplus(
            v @ self.W + self.h_bias
        ).sum(dim=1)


class DBN(nn.Module):
    """Deep Belief Network: stacked RBMs with a linear regression head.

    ``forward`` keeps the scaffold semantics (mean-field hidden probabilities
    propagated upward, then the classifier). ``pretrain`` / ``encode`` are
    additive and do not change that contract.
    """

    def __init__(self, layers=[64, 48, 32, 16, 8]):
        super().__init__()
        self.layers = list(layers)
        self.rbms = nn.ModuleList([RBM(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.classifier = nn.Linear(layers[-1], 1)

    def forward(self, x):
        for rbm in self.rbms:
            prob, _ = rbm.sample_h(x)
            x = prob
        return self.classifier(x)

    def encode(self, x):
        """Return the deep feature representation (without the classifier)."""
        for rbm in self.rbms:
            prob, _ = rbm.sample_h(x)
            x = prob
        return x

    def pretrain(self, x, epochs=PRETRAIN_EPOCHS, lr=PRETRAIN_LR,
                 batch_size=PRETRAIN_BATCH):
        """Greedy layer-wise CD-1 pretraining; returns per-RBM recon MSE.

        Classic DBN protocol: at the bottom layer the visible units see
        thresholded sensor bits (Bernoulli RBMs need contrasty inputs to
        learn detectors) and sampled hidden states are chained upward. The
        contrastive-divergence loop the original scaffold omitted — the
        wt-pm platform adapter ``m09-dbn-features`` supplies an equivalent
        loop externally; having one here keeps the repo self-contained.
        """
        errors = []
        v = x
        n = len(v)
        for depth, rbm in enumerate(self.rbms):
            opt = torch.optim.SGD(rbm.parameters(), lr=lr)
            visible = (v > 0.5).float() if depth == 0 else v
            for _ in range(epochs):
                perm = torch.randperm(n)
                for i in range(0, n, batch_size):
                    batch = visible[perm[i:i + batch_size]]
                    p_h, h = rbm.sample_h(batch)
                    p_v, v_neg = rbm.sample_v(h)
                    # CD-1: minimise F(v_data) - F(v_reconstructed)
                    loss = (rbm.free_energy(batch) - rbm.free_energy(v_neg)).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            with torch.no_grad():
                p_h, h = rbm.sample_h(visible)
                p_v, _ = rbm.sample_v(p_h)
                errors.append(float(torch.mean((visible - p_v) ** 2)))
                v = h  # sampled binary chaining into the next RBM
        return errors


# ----------------------------------------------------------------------------
# Synthetic fleet campaign (64-dim sensor features, turbine wear curves)
# ----------------------------------------------------------------------------
def generate_fleet():
    """Fleet of synthetic turbines observed along campaign time.

    Each turbine has its own wear exponent p and service life, and is
    inspected every INSPECTION_DAYS campaign days. Sensors report the wear
    state (1 - y)^p and its rate p·(1 - y)^(p-1) — the nonlinear pair a
    reader must invert to recover normalized RUL y — plus load/thermal
    oscillations and interactions, all in (0, 1) (the visible-probability
    range the RBMs expect). Returns (features, rul_norm, campaign_days).
    """
    rng = np.random.default_rng(SEED)
    loadings = rng.normal(0.0, 1.0, size=(8, 64))
    strengths = np.array([1.0, 1.0, 0.3, 0.3, 0.25, 0.25, 0.25, 0.5])
    loadings = loadings * strengths[:, None]

    days_list, latent_list, rul_list = [], [], []
    for _ in range(N_TURBINES):
        start = rng.uniform(*START_RANGE)
        life = rng.uniform(*LIFE_RANGE)
        p = rng.uniform(*WEAR_EXPONENT_RANGE)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        age = INSPECTION_DAYS
        while age < life:
            y = 1.0 - age / life                      # normalized RUL
            wear = (1.0 - y) ** p                     # wear/crack state
            rate = p * (1.0 - y) ** (p - 1.0)         # wear-rate channel
            load = np.sin(2 * np.pi * age / 30.0 + phase)
            thermal = np.sin(2 * np.pi * age / 11.0 + phase / 2.0)
            latent_list.append([wear, rate, load, thermal,
                                wear * load, wear * thermal, rate * load,
                                wear * rate])
            rul_list.append(y)
            days_list.append(start + age)
            age += INSPECTION_DAYS

    latent = np.asarray(latent_list)
    raw = latent @ loadings
    raw = raw + rng.normal(0.0, 0.18, size=raw.shape)
    days = np.asarray(days_list)
    rul = np.asarray(rul_list)

    order = np.argsort(days)                          # chronological order
    days, raw, rul = days[order], raw[order], rul[order]

    n = len(rul)
    n_train = int(n * (1.0 - VAL_FRACTION - TEST_FRACTION))
    mean = raw[:n_train].mean(axis=0)                 # train-window stats only
    std = raw[:n_train].std(axis=0) + 1e-9
    raw = (raw - mean) / std
    features = torch.clamp(torch.tensor(0.5 + raw / 8.0, dtype=torch.float32),
                           1e-3, 1.0 - 1e-3)          # affine ramp into (0, 1)
    return features, torch.tensor(rul, dtype=torch.float32), days


def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _report(dbn, features, rul_norm, lo, hi, tag):
    with torch.no_grad():
        pred = dbn(features[lo:hi]).squeeze(1).numpy()
    pred = np.clip(pred, 0.0, 1.0)
    y = rul_norm[lo:hi].numpy()
    return {
        "split": tag,
        "rmse_norm": float(np.sqrt(np.mean((pred - y) ** 2))),
        "rmse_days": float(np.sqrt(np.mean((pred - y) ** 2)) * MEAN_LIFE_DAYS),
        "mae_days": float(np.mean(np.abs(pred - y)) * MEAN_LIFE_DAYS),
        "r2": _r2(y, pred),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    features, rul_norm, days = generate_fleet()

    n = len(features)
    n_val = int(round(n * VAL_FRACTION))
    n_test = int(round(n * TEST_FRACTION))
    n_train = n - n_val - n_test
    print(f"Model 5 — 5-layer DBN | fleet={N_TURBINES} turbines, {n} inspections, "
          f"64 channels")
    print(f"Chronological split on campaign day "
          f"[{days[0]:.0f}..{days[-1]:.0f}]: {n_train}/{n_val}/{n_test}")

    dbn = DBN(layers=[features.shape[1], 48, 32, 16, 8])
    params = sum(p.numel() for p in dbn.parameters())
    print(f"Architecture: {dbn.layers} -> Linear(1)  |  {params:,} parameters")

    # 1) Greedy layer-wise CD-1 pretraining
    recon_errors = dbn.pretrain(features[:n_train])
    print("CD-1 pretraining recon MSE per RBM (no-skill ~0.25): "
          + ", ".join(f"{e:.4f}" for e in recon_errors))

    # 2) Supervised fine-tuning of the full stack (mean-field probabilities)
    opt = torch.optim.Adam([
        {"params": dbn.rbms.parameters(), "lr": STACK_LR},
        {"params": dbn.classifier.parameters(), "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    best_val, best_state, wait, epochs_done = float("inf"), None, 0, 0
    for epoch in range(FINE_TUNE_EPOCHS):
        dbn.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            loss = loss_fn(dbn(features[idx]).squeeze(1), rul_norm[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        epochs_done = epoch + 1
        dbn.eval()
        with torch.no_grad():
            val_pred = dbn(features[n_train:n_train + n_val]).squeeze(1)
        val_rmse = float(torch.sqrt(loss_fn(val_pred, rul_norm[n_train:n_train + n_val])))
        if val_rmse < best_val - 1e-5:
            best_val, wait = val_rmse, 0
            best_state = {k: v.clone() for k, v in dbn.state_dict().items()}
        else:
            wait += 1
            if wait >= EARLY_STOPPING_PATIENCE:
                break
    if best_state is not None:
        dbn.load_state_dict(best_state)
    dbn.eval()
    print(f"Fine-tuning: {epochs_done} epochs (stack lr {STACK_LR}, head lr {HEAD_LR}, "
          f"patience {EARLY_STOPPING_PATIENCE}), best val RMSE {best_val:.4f}")

    # 3) Evaluation
    metrics = [
        _report(dbn, features, rul_norm, n_train + n_val, n, "test"),
        _report(dbn, features, rul_norm, n_train, n_train + n_val, "val"),
        _report(dbn, features, rul_norm, 0, n_train, "train"),
    ]
    print(f"{'split':<6}{'rmse_norm':>11}{'rmse_days':>11}{'mae_days':>10}{'r2':>9}")
    for row in metrics:
        print(f"{row['split']:<6}{row['rmse_norm']:>11.4f}{row['rmse_days']:>11.1f}"
              f"{row['mae_days']:>10.1f}{row['r2']:>9.4f}")

    # 4) Persist
    torch.save(dbn.state_dict(), os.path.join(RESULTS_DIR, WEIGHTS_NAME))
    summary = {
        "model": "model-5-dbn",
        "layers": dbn.layers,
        "params": params,
        "pretrain": {
            "algorithm": "CD-1",
            "epochs_per_rbm": PRETRAIN_EPOCHS,
            "lr": PRETRAIN_LR,
            "batch_size": PRETRAIN_BATCH,
            "visible_input": "thresholded sensor bits at depth 0; sampled binary chaining upward",
            "recon_mse_per_rbm": [round(e, 4) for e in recon_errors],
        },
        "fine_tune": {
            "optimizer": "adam",
            "stack_lr": STACK_LR,
            "head_lr": HEAD_LR,
            "weight_decay": WEIGHT_DECAY,
            "loss": "mse",
            "epochs_completed": epochs_done,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        },
        "data": {
            "fleet_turbines": N_TURBINES,
            "inspections": n,
            "features": int(features.shape[1]),
            "target": "normalized RUL",
            "split": "chronological campaign-day 65/15/20",
            "day_range": [float(days[0]), float(days[-1])],
        },
        "metrics": metrics,
    }
    with open(os.path.join(RESULTS_DIR, METRICS_NAME), "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    print(f"Wrote {RESULTS_DIR}/{METRICS_NAME} and {RESULTS_DIR}/{WEIGHTS_NAME}")


if __name__ == "__main__":
    main()
