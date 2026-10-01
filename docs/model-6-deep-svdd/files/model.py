"""Model 6 — 6-layer Deep SVDD one-class boundary for SCADA anomaly detection.

Extends the original scaffold (3-layer MLP with biases, no data, no training
loop) to a **6-layer bias-free encoder** trained with the one-class Deep SVDD
objective (Ruff et al., ICML 2018):

    Linear(in→128) → Linear(128→96) → Linear(96→64)
        → Linear(64→48) → Linear(48→32) → Linear(32→rep_dim)   [no biases]
    LeakyReLU(0.1) between layers · score = ‖φ(x) − c‖²

Why bias-free: with bias terms the network can map every input to the centre
``c`` (constant output, zero loss) — the "hypersphere collapse" trivial
solution. Removing biases (and keeping ``c`` fixed after initialisation) rules
that out, as in the original Deep SVDD paper.

Pipeline
--------
1. Synthesize hourly SCADA telemetry for a 24-turbine farm over 120 days:
   16 physical channels (wind, power, rotor speed, pitch, yaw error,
   temperatures with first-order thermal lag, vibrations, phase-current
   imbalance) sampled every 10 minutes and summarized per hour as
   mean / std / min / max → 64 features.
2. Inject six incipient fault types into 12 turbines during the test window
   only (gearbox bearing overheating, generator winding hot-spot, pitch
   misalignment, yaw misalignment, drivetrain bearing wear, rotor imbalance),
   each ramping in over 4–10 days.
3. Split **chronologically**: days 0–70 train (healthy only — one-class),
   days 70–85 validation (healthy; early stopping + alarm threshold),
   days 85–120 test (healthy + faulty turbines).
4. Standardize, then ZCA-whiten with statistics from the healthy training
   window (decorrelates the mean/std/min/max views of each channel so no
   single physical signal dominates the distance). Initialise the centre with
   ``init_center`` and train with ``svdd_loss`` for a fixed, short schedule
   (20 epochs, Adam lr 1e-4) — Deep SVDD keeps lowering healthy-data loss
   long after fault contrast stops improving, and with healthy-only
   validation there is no label-free signal to stop on, so the budget is
   fixed up front.
5. Alarm threshold = 99th percentile of validation distances; an alarm is
   raised after 3 consecutive hourly exceedances. Report AUROC, point-wise
   precision/recall/F1, false-alarm rate, per-fault detection delay — and the
   identical protocol run on the original 3-layer scaffold network, plus a
   linear Mahalanobis-distance reference.

Run:
    python model.py              # seed 42 -> results/model6_metrics.json
    python model.py --seed 3     # any other seed (farm + init are regenerated)

The wt-pm platform imports this file through adapter ``m13-deep-svdd`` and
calls ``DeepSVDDNetwork(input_dim=..., rep_dim=16)``,
``init_center(model, [(Xs,)])`` and ``svdd_loss(model(Xs), model.c)`` — keep
those names and signatures stable. Importing this module has no side effects
(no seeding, no training); everything below ``main()`` runs only as a script.
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SEED = 42
N_TURBINES = 24
DAYS = 120
SUBSTEPS = 6                     # 10-minute samples per hourly record
TRAIN_END_DAY = 70               # [0, 70)   healthy — training
VAL_END_DAY = 85                 # [70, 85)  healthy — early stopping + threshold
FAULT_ONSET_RANGE = (88.0, 108.0)  # fault onsets (campaign days), test window only
FAULT_RAMP_RANGE = (4.0, 10.0)     # days for a fault to reach full severity
FAULT_TYPES = (
    "gearbox_bearing", "generator_winding", "pitch_misalignment",
    "yaw_misalignment", "drivetrain_bearing", "rotor_imbalance",
)
TURBINES_PER_FAULT = 2           # 6 types x 2 = 12 faulty, 12 healthy turbines

HIDDEN = (128, 96, 64, 48, 32)   # 5 hidden widths + rep layer = 6 linear layers
REP_DIM = 32
LEAKY_SLOPE = 0.1
SVDD_EPOCHS = 20                 # fixed budget (see docstring, step 4)
SVDD_LR = 1e-4
WEIGHT_DECAY = 1e-6
BATCH_SIZE = 256
WHITEN_SHRINK = 1e-2             # eigenvalue floor for ZCA whitening
SENSITIVITY_EPOCHS = (5, 10, 20, 30, 40)   # diagnostic only, never used for selection
THRESHOLD_QUANTILE = 0.99
ALARM_PERSISTENCE_H = 3
RESULTS_DIR = "results"
WEIGHTS_NAME = "model6_weights.pt"
METRICS_NAME = "model6_metrics.json"

CHANNELS = (
    "wind_speed", "power", "rotor_speed", "pitch_angle", "yaw_error",
    "ambient_temp", "nacelle_temp", "gearbox_oil_temp", "gearbox_bearing_temp",
    "gen_bearing_temp", "gen_winding_u", "gen_winding_v", "gen_winding_w",
    "drivetrain_vib", "tower_vib", "current_imbalance",
)
STATS = ("mean", "std", "min", "max")
FEATURE_NAMES = tuple(f"{c}_{s}" for s in STATS for c in CHANNELS)


# ----------------------------------------------------------------------------
# Model 6 architecture (platform API: DeepSVDDNetwork / init_center / svdd_loss)
# ----------------------------------------------------------------------------
class DeepSVDDNetwork(nn.Module):
    """6-layer bias-free Deep SVDD encoder φ(x) → R^rep_dim.

    ``model.c`` holds the hypersphere centre once ``init_center`` has run.
    """

    def __init__(self, input_dim=64, rep_dim=REP_DIM, hidden=HIDDEN):
        super().__init__()
        self.input_dim = input_dim
        self.rep_dim = rep_dim
        dims = [input_dim, *hidden, rep_dim]
        layers = []
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(d_in, d_out, bias=False))
            if i < len(dims) - 2:
                layers.append(nn.LeakyReLU(LEAKY_SLOPE))
        self.net = nn.Sequential(*layers)
        # Variance-preserving init so representations keep their scale through
        # six layers (PyTorch's default init shrinks them ~3x per layer).
        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        for lin in linears[:-1]:
            nn.init.kaiming_normal_(lin.weight, a=LEAKY_SLOPE, nonlinearity="leaky_relu")
        nn.init.xavier_normal_(linears[-1].weight)
        self.c = None  # hypersphere centre

    def forward(self, x):
        return self.net(x)


def init_center(model, loader, device="cpu", eps=0.1):
    """Set ``model.c`` to the mean representation over ``loader``.

    ``loader`` yields 1-tuples ``(x,)`` (a DataLoader over a TensorDataset, or
    simply ``[(X,)]``). Coordinates with magnitude below ``eps`` are pushed to
    ±``eps`` so the centre is never at (or near) the origin, which a bias-free
    network could reach trivially. Returns the centre as well.
    """
    model.eval()
    total, n = None, 0
    with torch.no_grad():
        for x, in loader:
            z = model(x.to(device))
            s = z.sum(dim=0)
            total = s if total is None else total + s
            n += z.shape[0]
    c = total / n
    c[(c.abs() < eps) & (c < 0)] = -eps
    c[(c.abs() < eps) & (c >= 0)] = eps
    model.c = c
    return c


def svdd_loss(outputs, c):
    """One-class Deep SVDD objective: mean squared distance to the centre."""
    return torch.mean(torch.sum((outputs - c) ** 2, dim=1))


def anomaly_score(model, x):
    """Squared distance ‖φ(x) − c‖² per sample (additive helper)."""
    model.eval()
    with torch.no_grad():
        return ((model(x) - model.c) ** 2).sum(dim=1)


def train_svdd(model, X_train, X_val=None, epochs=SVDD_EPOCHS, lr=SVDD_LR,
               batch_size=BATCH_SIZE, generator=None, on_epoch=None):
    """Mini-batch Deep SVDD training for a fixed number of epochs (additive helper).

    ``model.c`` must already be set (``init_center``) and stays fixed.
    ``on_epoch(epoch, model)`` is an optional callback (diagnostics only).
    Returns a small history dict with train/val loss per epoch.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    hist = {"train_loss": [], "val_loss": []}
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(X_train.shape[0], generator=generator)
        for i in range(0, X_train.shape[0], batch_size):
            xb = X_train[perm[i:i + batch_size]]
            opt.zero_grad()
            loss = svdd_loss(model(xb), model.c)
            loss.backward()
            opt.step()
        hist["train_loss"].append(float(anomaly_score(model, X_train).mean()))
        if X_val is not None:
            hist["val_loss"].append(float(anomaly_score(model, X_val).mean()))
        if on_epoch is not None:
            on_epoch(epoch, model)
    return hist


# ----------------------------------------------------------------------------
# Synthetic SCADA farm with incipient faults
# ----------------------------------------------------------------------------
def _power_curve(v):
    p = np.clip((v - 3.0) / 9.0, 0.0, 1.0) ** 3
    return np.where(v >= 25.0, 0.0, p)


def generate_farm(rng):
    """Return hourly features (N_TURBINES, hours, 64), fault severity and fault map."""
    steps_per_day = 24 * SUBSTEPS
    T = DAYS * steps_per_day
    nt = N_TURBINES
    t_day = np.arange(T) / steps_per_day                      # campaign day of each 10-min step
    hour = (t_day % 1.0) * 24.0

    # --- fault assignment (test window only) --------------------------------
    order = rng.permutation(nt)
    fault_of = {}
    for k, ftype in enumerate(FAULT_TYPES):
        for j in range(TURBINES_PER_FAULT):
            fault_of[int(order[k * TURBINES_PER_FAULT + j])] = ftype
    onset = np.full(nt, np.inf)
    ramp = np.ones(nt)
    for tid in fault_of:
        onset[tid] = rng.uniform(*FAULT_ONSET_RANGE)
        ramp[tid] = rng.uniform(*FAULT_RAMP_RANGE)
    sev = np.clip((t_day[None, :] - onset[:, None]) / ramp[:, None], 0.0, 1.0)  # (nt, T)

    def is_(ftype):
        return np.array([fault_of.get(i) == ftype for i in range(nt)], float)[:, None]

    # --- wind ----------------------------------------------------------------
    seasonal = 2.0 * np.sin(2 * np.pi * t_day / 37.0) + 1.0 * np.sin(2 * np.pi * t_day / 11.0)
    diurnal = 0.8 * np.sin(2 * np.pi * (hour - 14.0) / 24.0)
    farm = np.zeros(T)
    turb = np.zeros((nt, T))
    e_f = rng.normal(0, 0.30, T)
    e_t = rng.normal(0, 0.35, (nt, T))
    for t in range(1, T):
        farm[t] = 0.985 * farm[t - 1] + e_f[t]
        turb[:, t] = 0.90 * turb[:, t - 1] + e_t[:, t]
    v = np.clip(7.8 + seasonal + diurnal + 1.2 * farm + turb, 0.0, 28.0)

    # --- turbine-specific calibration ---------------------------------------
    eff = rng.uniform(0.96, 1.04, (nt, 1))
    t_off = rng.normal(0.0, 1.0, (nt, 1))
    vib_gain = rng.uniform(0.9, 1.1, (nt, 1))

    # --- yaw / pitch / power -------------------------------------------------
    yaw = np.zeros((nt, T))
    e_y = rng.normal(0, 1.2, (nt, T))
    for t in range(1, T):
        yaw[:, t] = 0.95 * yaw[:, t - 1] + e_y[:, t]
    yaw = yaw + 12.0 * sev * is_("yaw_misalignment")
    pitch = np.where(v > 11.5, 2.2 * np.clip(v - 11.5, 0, None) ** 0.9, 0.0)
    pitch = pitch + 3.0 * sev * is_("pitch_misalignment") + rng.normal(0, 0.15, (nt, T))
    p = eff * _power_curve(v) * np.cos(np.deg2rad(yaw)) ** 3
    p = p * (1.0 - 0.15 * sev * is_("pitch_misalignment") * (v > 9.0))
    p = p * (1.0 - 0.10 * sev * is_("rotor_imbalance"))
    p = np.clip(p + rng.normal(0, 0.015, (nt, T)), 0.0, 1.05)
    rotor = np.where(v >= 3.0, np.minimum(6.0 + 1.25 * (v - 3.0), 16.0), 2.0)
    rotor = rotor + rng.normal(0, 0.15, (nt, T)) * (1.0 + 2.0 * sev * is_("rotor_imbalance"))

    # --- temperatures (first-order thermal lag) ------------------------------
    amb = (9.0 + 5.0 * np.sin(2 * np.pi * t_day / 120.0) + 3.5 * np.sin(2 * np.pi * (hour - 15.0) / 24.0)
           + rng.normal(0, 0.3, (nt, T)) + t_off)
    s_gb = sev * is_("gearbox_bearing")
    s_gw = sev * is_("generator_winding")
    s_db = sev * is_("drivetrain_bearing")
    targets = {
        "nacelle": amb + 8.0 + 6.0 * p,
        "oil": amb + 20.0 + 25.0 * p + 4.0 * s_db,
        "gen_bear": amb + 15.0 + 18.0 * p,
        "wind_u": amb + 25.0 + 45.0 * p ** 1.5 + 15.0 * s_gw * (0.3 + p),
        "wind_v": amb + 25.0 + 45.0 * p ** 1.5,
        "wind_w": amb + 25.0 + 45.0 * p ** 1.5,
    }
    taus = {"nacelle": 18, "oil": 12, "gen_bear": 9, "wind_u": 4, "wind_v": 4, "wind_w": 4}
    temps = {k: np.empty((nt, T)) for k in targets}
    for k in targets:
        temps[k][:, 0] = targets[k][:, 0]
    bear = np.empty((nt, T))
    bear_target_extra = 12.0 * s_gb * (0.4 + p)
    bear[:, 0] = temps["oil"][:, 0] + 5.0 + 10.0 * p[:, 0]
    for t in range(1, T):
        for k, tau in taus.items():
            temps[k][:, t] = temps[k][:, t - 1] + (targets[k][:, t] - temps[k][:, t - 1]) / tau
        tb = temps["oil"][:, t] + 5.0 + 10.0 * p[:, t] + bear_target_extra[:, t]
        bear[:, t] = bear[:, t - 1] + (tb - bear[:, t - 1]) / 6.0
    noise_t = lambda: rng.normal(0, 0.25, (nt, T))  # noqa: E731

    # --- vibration / electrical ----------------------------------------------
    dvib = vib_gain * (0.3 + 0.08 * rotor + 0.4 * p) * rng.lognormal(0, 0.08, (nt, T))
    dvib = dvib * (1.0 + 1.5 * s_db)
    tvib = (0.05 + 0.02 * v) * rng.lognormal(0, 0.10, (nt, T)) * (1.0 + 1.2 * sev * is_("rotor_imbalance"))
    imb = np.abs(0.5 + rng.normal(0, 0.15, (nt, T))) + 3.0 * s_gw

    raw = np.stack([
        v, p, rotor, pitch, yaw, amb,
        temps["nacelle"] + noise_t(), temps["oil"] + noise_t(), bear + noise_t(),
        temps["gen_bear"] + noise_t(), temps["wind_u"] + noise_t(),
        temps["wind_v"] + noise_t(), temps["wind_w"] + noise_t(),
        dvib, tvib, imb,
    ], axis=-1)                                              # (nt, T, 16)

    hours = DAYS * 24
    r = raw.reshape(nt, hours, SUBSTEPS, len(CHANNELS))
    feats = np.concatenate([r.mean(2), r.std(2), r.min(2), r.max(2)], axis=-1).astype(np.float32)
    sev_h = sev.reshape(nt, hours, SUBSTEPS).mean(2)
    return feats, sev_h, fault_of, onset, ramp


# ----------------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------------
def _auroc(y, s):
    """Rank-based ROC AUC (Mann-Whitney U), ties averaged."""
    y = np.asarray(y, bool)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    s_sorted = np.asarray(s)[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos, n_neg = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _persistent(alarm, k=ALARM_PERSISTENCE_H):
    """Per-turbine alarm that fires once ``k`` consecutive hours exceed the threshold."""
    out = np.zeros_like(alarm)
    for i in range(alarm.shape[0]):
        run = 0
        for t in range(alarm.shape[1]):
            run = run + 1 if alarm[i, t] else 0
            out[i, t] = run >= k
    return out


def _evaluate(name, scores_test, thr, sev_test, test_day0, fault_of, onset, ramp):
    """Point-wise and event-level metrics on the test window."""
    y = sev_test > 0.0
    s = scores_test.ravel()
    exceed = scores_test > thr
    alarm = _persistent(exceed)
    tp = int((alarm & y).sum())
    fp = int((alarm & ~y).sum())
    fn = int((~alarm & y).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    healthy = [i for i in range(scores_test.shape[0]) if i not in fault_of]
    healthy_days = len(healthy) * scores_test.shape[1] / 24.0
    # alarm *events* on healthy turbines: rising edges of the persistent alarm
    fa_events = sum(int(np.sum(np.diff(alarm[i].astype(int), prepend=0) == 1)) for i in healthy)
    events = {}
    for tid, ftype in sorted(fault_of.items()):
        onset_h = int(np.ceil((onset[tid] - test_day0) * 24.0))
        hits = np.flatnonzero(alarm[tid, onset_h:])
        delay_h = float(hits[0]) if hits.size else None
        events.setdefault(ftype, []).append({
            "turbine": tid,
            "onset_day": round(float(onset[tid]), 2),
            "ramp_days": round(float(ramp[tid]), 2),
            "detected": delay_h is not None,
            "delay_hours": delay_h,
            "severity_at_detection": (round(float(sev_test[tid, onset_h + int(delay_h)]), 3)
                                      if delay_h is not None else None),
        })
    delays = [e["delay_hours"] for ev in events.values() for e in ev if e["detected"]]
    n_events = sum(len(ev) for ev in events.values())
    return {
        "model": name,
        "auroc": round(_auroc(y.ravel(), s), 4),
        "threshold": round(float(thr), 6),
        "pointwise": {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)},
        "healthy_turbine_hourly_fpr": round(float(alarm[healthy].mean()), 5),
        "false_alarm_events_per_100_turbine_days": round(100.0 * fa_events / healthy_days, 3),
        "events_detected": f"{len(delays)}/{n_events}",
        "median_detection_delay_hours": float(np.median(delays)) if delays else None,
        "events": events,
    }


class _ScaffoldSVDD(nn.Module):
    """The original 3-layer scaffold network (with biases), for comparison only."""

    def __init__(self, input_dim=64, rep_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.rep_dim = rep_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, rep_dim),
        )
        self.c = None

    def forward(self, x):
        return self.net(x)


def _fit_whitener(X):
    """ZCA whitening matrix from healthy training data (eigenvalue floor WHITEN_SHRINK)."""
    w, V = np.linalg.eigh(np.cov(X, rowvar=False))
    return ((V / np.sqrt(w + WHITEN_SHRINK)) @ V.T).astype(np.float32)


def _run(model, X_tr, X_va, X_te, te_shape, y_te, seed):
    """Fixed-budget Deep SVDD run; returns history, threshold, test scores, sensitivity."""
    g = torch.Generator().manual_seed(seed)
    init_center(model, torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr), batch_size=1024))
    snapshot = {}

    def at_epoch(epoch, m):
        if epoch == SVDD_EPOCHS:
            snapshot["state"] = {k: v.detach().clone() for k, v in m.state_dict().items()}
        if epoch in SENSITIVITY_EPOCHS:
            snapshot.setdefault("auroc_by_epoch", {})[epoch] = round(
                _auroc(y_te, anomaly_score(m, X_te).numpy()), 4)

    hist = train_svdd(model, X_tr, X_va, epochs=max(SVDD_EPOCHS, *SENSITIVITY_EPOCHS),
                      generator=g, on_epoch=at_epoch)
    model.load_state_dict(snapshot["state"])           # the pre-committed budget
    thr = float(np.quantile(anomaly_score(model, X_va).numpy(), THRESHOLD_QUANTILE))
    scores = anomaly_score(model, X_te).numpy().reshape(te_shape)
    summary = {
        "epochs": SVDD_EPOCHS,
        "train_loss": round(hist["train_loss"][SVDD_EPOCHS - 1], 5),
        "val_loss": round(hist["val_loss"][SVDD_EPOCHS - 1], 5),
    }
    return summary, thr, scores, snapshot["auroc_by_epoch"]


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    feats, sev, fault_of, onset, ramp = generate_farm(rng)
    h_tr, h_va = TRAIN_END_DAY * 24, VAL_END_DAY * 24
    assert sev[:, :h_va].max() == 0.0, "faults must stay out of train/val windows"
    nf = feats.shape[-1]
    tr = feats[:, :h_tr].reshape(-1, nf)
    va = feats[:, h_tr:h_va].reshape(-1, nf)
    te = feats[:, h_va:].reshape(-1, nf)
    te_shape = feats[:, h_va:].shape[:2]
    sev_te = sev[:, h_va:]
    y_te = (sev_te > 0).ravel()

    # preprocessing fitted on the healthy training window only
    mu, sd = tr.mean(0), tr.std(0) + 1e-6
    W = _fit_whitener((tr - mu) / sd)
    prep = lambda a: torch.tensor(((a - mu) / sd) @ W, dtype=torch.float32)  # noqa: E731
    X_tr, X_va, X_te = prep(tr), prep(va), prep(te)

    print(f"Farm: {N_TURBINES} turbines x {DAYS} days hourly -> {nf} features")
    print(f"Train {X_tr.shape[0]:,} | Val {X_va.shape[0]:,} | Test {X_te.shape[0]:,} records "
          f"({int(y_te.sum()):,} faulty)")

    torch.manual_seed(SEED)
    model = DeepSVDDNetwork(input_dim=nf, rep_dim=REP_DIM)
    n_params = sum(p.numel() for p in model.parameters())
    hist6, thr6, sc6, sens6 = _run(model, X_tr, X_va, X_te, te_shape, y_te, SEED)
    m6 = _evaluate("Model 6 (6-layer, bias-free)", sc6, thr6, sev_te,
                   VAL_END_DAY, fault_of, onset, ramp)

    torch.manual_seed(SEED)
    base = _ScaffoldSVDD(input_dim=nf, rep_dim=REP_DIM)
    histb, thrb, scb, sensb = _run(base, X_tr, X_va, X_te, te_shape, y_te, SEED)
    mb = _evaluate("original scaffold (3-layer, biases)", scb, thrb, sev_te,
                   VAL_END_DAY, fault_of, onset, ramp)

    # linear reference: Mahalanobis distance == squared norm in whitened space
    md = lambda X: (X ** 2).sum(1).numpy()  # noqa: E731
    mm = _evaluate("Mahalanobis distance (linear reference)", md(X_te).reshape(te_shape),
                   float(np.quantile(md(X_va), THRESHOLD_QUANTILE)), sev_te,
                   VAL_END_DAY, fault_of, onset, ramp)

    torch.save({"state_dict": model.state_dict(), "c": model.c, "mu": mu, "sd": sd,
                "whitener": W, "threshold": thr6, "feature_names": FEATURE_NAMES},
               os.path.join(RESULTS_DIR, WEIGHTS_NAME))

    print("\n{:<42} {:>7} {:>7} {:>7} {:>7} {:>8} {:>8}".format(
        "model", "AUROC", "prec", "recall", "F1", "events", "delay_h"))
    for m in (m6, mb, mm):
        pw = m["pointwise"]
        print("{:<42} {:>7.3f} {:>7.3f} {:>7.3f} {:>7.3f} {:>8} {:>8}".format(
            m["model"], m["auroc"], pw["precision"], pw["recall"], pw["f1"],
            m["events_detected"], m["median_detection_delay_hours"]))
    print(f"\nFalse-alarm events per 100 healthy turbine-days: Model 6 "
          f"{m6['false_alarm_events_per_100_turbine_days']} | scaffold "
          f"{mb['false_alarm_events_per_100_turbine_days']} | Mahalanobis "
          f"{mm['false_alarm_events_per_100_turbine_days']}")
    print("\nModel 6 per-fault detection delay (hours after onset):")
    for ftype, evs in m6["events"].items():
        print(f"  {ftype:<20} " + ", ".join(
            f"T{e['turbine']:02d}: {e['delay_hours']:.0f}h (sev {e['severity_at_detection']:.2f})"
            if e["detected"] else f"T{e['turbine']:02d}: missed" for e in evs))
    print(f"\nAUROC vs epochs (diagnostic, not used for selection): Model 6 {sens6} | scaffold {sensb}")

    metrics = {
        "model": "model-6-deep-svdd",
        "architecture": {
            "layers": [nf, *HIDDEN, REP_DIM],
            "linear_layers": len(HIDDEN) + 1,
            "bias": False,
            "activation": f"LeakyReLU({LEAKY_SLOPE})",
            "params": n_params,
        },
        "training": {
            "preprocessing": f"standardize + ZCA whitening (eigenvalue floor {WHITEN_SHRINK}), "
                             "fitted on the healthy training window",
            "centre": "init_center: mean training representation, |c_i| >= 0.1",
            "svdd": {"optimizer": "adam", "lr": SVDD_LR, "weight_decay": WEIGHT_DECAY,
                     "batch_size": BATCH_SIZE, **hist6},
            "threshold": f"{THRESHOLD_QUANTILE:.0%} quantile of healthy validation distances",
            "alarm_persistence_hours": ALARM_PERSISTENCE_H,
        },
        "data": {
            "turbines": N_TURBINES, "days": DAYS, "records": int(feats.shape[0] * feats.shape[1]),
            "features": nf, "feature_layout": "16 channels x (mean, std, min, max) per hour",
            "split": f"chronological: train days 0-{TRAIN_END_DAY} (healthy), "
                     f"val {TRAIN_END_DAY}-{VAL_END_DAY} (healthy), test {VAL_END_DAY}-{DAYS}",
            "faulty_turbines": len(fault_of), "fault_types": list(FAULT_TYPES),
            "test_faulty_records": int(y_te.sum()), "test_records": int(y_te.size),
            "fault_label": "severity > 0 (includes the earliest, near-invisible part of each ramp)",
        },
        "test": m6,
        "auroc_by_epoch_diagnostic": {"model6": sens6, "scaffold": sensb},
        "baseline_scaffold": {**mb, "training": histb,
                              "params": sum(p.numel() for p in base.parameters())},
        "reference_mahalanobis": mm,
        "seed": SEED,
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(RESULTS_DIR, METRICS_NAME), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved {RESULTS_DIR}/{METRICS_NAME} and {RESULTS_DIR}/{WEIGHTS_NAME} "
          f"({metrics['runtime_s']}s)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Model 6 Deep SVDD: synthesize, train, evaluate")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    args = ap.parse_args()
    SEED, RESULTS_DIR = args.seed, args.results_dir
    main()
