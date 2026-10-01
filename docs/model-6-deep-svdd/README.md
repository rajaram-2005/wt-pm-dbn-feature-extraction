# Model 6 — 6-layer Deep SVDD (staged for `wt-pm-deep-svdd-boundary`)

Built and verified from the dbn-feature-extraction session. That session's GitHub
access can't push to `wt-pm-deep-svdd-boundary`, so the finished change is staged here.
It applies cleanly on deep-svdd `main` at `b0c13e7`.

| path | what it is |
|---|---|
| `0001-model-6-deep-svdd.patch` | The whole change as one commit (`git am`) |
| `files/` | The same files, ready to upload: `model.py`, `README.md`, `requirements.txt`, `index.html`, `docs/index.html`, `results/model6_metrics.json`, `results/model6_seed_sweep.json` |

## What Model 6 is

The original starter code was a 3-layer MLP with biases, with no data and no training loop.
Model 6 replaces it with a **6-layer bias-free Deep SVDD** (64→128→96→64→48→32→rep,
LeakyReLU, Kaiming init; 32,256 params). It adds a synthetic 24-turbine SCADA farm with six
incipient fault types, a chronological healthy-only train/val split, ZCA whitening, a
fixed 20-epoch schedule, and alarm-level evaluation against the old network and a
Mahalanobis reference. The full write-up is in `files/README.md`.

| 6 seeds | AUROC | F1 | events detected | median delay | false alarms /100 turbine-days |
|---|---|---|---|---|---|
| **Model 6** | **0.885 ± 0.016** | **0.782** | **68/72** | **56.5 h** | 0.04 |
| original scaffold | 0.864 ± 0.013 | 0.702 | 59/72 | 70.0 h | 0.04 |
| Mahalanobis | 0.877 ± 0.009 | 0.661 | 63/72 | 95.0 h | 1.03 |

**Platform:** the `m13-deep-svdd` adapter API is unchanged (`DeepSVDDNetwork(input_dim, rep_dim=16)`,
`init_center(model, [(Xs,)])`, `svdd_loss(model(Xs), model.c)`). The platform's
`test_deep_svdd_detects_ramp` passes with the new file.

## How to apply

**Option A: GitHub website (no terminal)**
1. Open https://github.com/rajaram-2005/wt-pm-deep-svdd-boundary
2. Click **Add file → Upload files**.
3. Drag in the *contents* of `files/` (`model.py`, `README.md`, `requirements.txt`,
   `index.html`, and the `docs/` and `results/` folders), keeping the folder structure.
4. Commit to `main` with the message `Model 6: 6-layer bias-free Deep SVDD`.

**Option B: terminal**
```bash
git clone https://github.com/rajaram-2005/wt-pm-deep-svdd-boundary.git && cd wt-pm-deep-svdd-boundary
curl -sL https://raw.githubusercontent.com/rajaram-2005/wt-pm-dbn-feature-extraction/main/docs/model-6-deep-svdd/0001-model-6-deep-svdd.patch | git am
pip install -r requirements.txt && python model.py   # check: "Model 6 ... 0.865 ... 12/12"
git push origin main
```
(Before this folder is merged to `main`, replace `main` in the URL with
`arena/01a0f6ff-wt-pm-dbn-feature-extraction`.)

**After applying:** `model.py`'s sha256 changes, so the deep-svdd entry in
`wt-pm-lstm-scada-anomaly/deployment/models.lock.json` must be re-pinned. Run the
re-pin script in `docs/audit-models-1-4-fixes.md` (Fix 3). It updates every entry that has drifted.
