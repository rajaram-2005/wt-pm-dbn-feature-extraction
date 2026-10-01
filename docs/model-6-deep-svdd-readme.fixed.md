# DEEP-SVDD-BOUNDARY
> **Category:** SCADA Anomaly Detection  
> **Platform adapter:** `m13-deep-svdd`  
> **Status:** integrated

## Description
Deep Support Vector Data Description mapping telemetry into a hypersphere boundary for abnormal state isolation.

This is model **m13** of the WT-PM wind-turbine predictive-maintenance suite. The repository stays independently usable (`python model.py`); the unified **wt-pm** platform wraps the original `model.py` through adapter `m13-deep-svdd` — it does not replace or fork this code.

The repository ships the SVDD building blocks — the feature-space network, the hypersphere-centre initialiser and the SVDD objective. The platform adapter supplies the training loop: it standardizes its feature table, initializes the centre with the repo's `init_center`, then minimizes `svdd_loss` for 60 Adam steps. Anomaly score at inference is the squared distance of a sample's representation to the learned centre.

```
Input (input_dim standardized features)
  └─ Linear(input_dim → 64) → ReLU
  └─ Linear(64 → 32) → ReLU
  └─ Linear(32 → rep_dim)          # hypersphere representation z
      svdd_loss = mean ‖z − c‖²    # c = hypersphere centre (init_center)
```

## Structure
```
├── model.py           # DeepSVDDNetwork, init_center, svdd_loss (definitions only)
├── requirements.txt   # Dependencies
├── .gitignore         # Environment and data exclusions
└── README.md          # Project overview
```

## Setup & Execution
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Inspect the module (definitions only — there is no training script or dataset in this repository):
   ```bash
   python model.py
   ```
   `model.py` defines `DeepSVDDNetwork`, `init_center` and `svdd_loss` and exits; training is performed by the caller (the platform adapter, or your own loop).

## Model

[`model.py`](model.py) defines:

| Block | Definition |
|---|---|
| `DeepSVDDNetwork(input_dim=64, rep_dim=32)` | `Sequential(Linear(input_dim, 64), ReLU, Linear(64, 32), ReLU, Linear(32, rep_dim))`; `forward(x)` returns the representation `z`; `model.c` holds the hypersphere centre once initialised |
| `init_center(model, loader, device="cpu", eps=0.1)` | One forward pass over `loader` (batches as 1-tuples `(x,)`); sets `model.c` to the mean representation, clipped away from zero by `eps` |
| `svdd_loss(outputs, c)` | Mean over the batch of squared distances `‖z − c‖²` |

- **Input:** feature tensor `x` of shape `(batch_size, input_dim)` (`float32`).
- **Output:** representation `z` of shape `(batch_size, rep_dim)`; anomaly score is `‖z − c‖²` computed by the caller.
- **Size:** 7,296 parameters for the default `(input_dim=64, rep_dim=32)` configuration (5,104 parameters when the platform instantiates it on its 38-column feature table with `rep_dim=16`).

`requirements.txt` lists `torch numpy scikit-learn`; `model.py` itself only imports `torch` (`torch.nn`). Verified with Python 3.11 and PyTorch 2.14.

## Platform contract

| | |
|---|---|
| Input (adapter view) | tabular features (`batch.features`) |
| Output (`ModelOutput` / `WTDataSchema`) | `anomaly_score` — robust-calibrated distance to the hypersphere centre |
| Integration role | One-class anomaly stream (`TaskType.ANOMALY_DETECTION`), routed as a Hermes `anomaly` voter |
| Deployment / fallback | `cloud`, `edge_gpu`, `edge_cpu` (requires `torch`) · falls back to `m14-isolation-forest` |

The platform adapter `m13-deep-svdd` calls the original API surface exactly as defined here: `DeepSVDDNetwork(input_dim=..., rep_dim=16)`, `init_center(model, [(Xs,)])` (note the 1-tuple batches), and `svdd_loss(model(Xs), model.c)` — 60 Adam steps (lr 1e-3, weight decay 1e-5) on sigmoid/robust-standardized features. Records emitted via `ModelOutput.to_records()` follow `WTDataSchema` (`wt-pm.platform.v1`).

To run this model inside the platform (Python ≥ 3.10):

```bash
pip install "wt-pm[torch] @ git+https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly.git#subdirectory=platform"
wt-pm fetch-models --dir ./external      # clones this repo and its siblings into ./external
export WTPM_EXTERNAL_DIR=$PWD/external   # adapters import external/<repo>/model.py
wt-pm inspect                            # m13-deep-svdd should report "available": true
```

More on how the 25 models fit together: [platform architecture](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/ARCHITECTURE.md) and the [per-model table](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/platform/docs/MODELS.md).

## Model Info
- **Repo name:** `wt-pm-deep-svdd-boundary`
- **Category:** SCADA Anomaly Detection
- **Model:** Deep SVDD MLP encoder (~7.3K parameters default / ~5.1K in platform configuration)
- **Dependencies:** `torch` (`numpy`, `scikit-learn` listed for caller convenience)

## Honest limits

- Definitions only: no dataset, training loop, or trained weights ship in this repository; the platform adapter trains on standardized simulator features (fidelity rung 1, see the [fidelity ladder](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/docs/fidelity_ladder.md)).
- One-class boundary quality depends on the training window being genuinely healthy; contamination shifts the centre toward faults.
- `init_center` clips centre coordinates with magnitude below `eps=0.1` to ±`eps` (standard Deep-SVDD practice to avoid trivial solutions at the origin).
- Anomaly scores are calibrated by the platform adapter (robust z), not by this repository.
