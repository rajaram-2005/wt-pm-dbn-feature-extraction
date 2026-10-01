# Audit — Model 6 (`wt-pm-deep-svdd-boundary`, model id m13)

**Date:** 2026-10-01 · **Continues:** [audit-models-1-4-fixes.md](audit-models-1-4-fixes.md)
**Method:** same as the models 1–4 audit — adapter contract checked against `model.py` at `main` HEAD, README claims recomputed, behaviour verified live.

## Suite-wide contract sweep (done as part of this pass)

Every `mod.<symbol>` call in all four adapter modules (`anomaly`, `classification`, `physics_graph_edge`, `prognostics`) was extracted and checked against the corresponding repo's `model.py` at its current `main` HEAD — all 25 models:

- **Broken contracts: exactly one** — `build_convlstm` (m02, `wt-pm-convlstm-wear-prognostics`), already covered by Fix 1 of the models 1–4 audit.
- All other symbol contracts pass, including m13: `DeepSVDDNetwork`, `init_center`, `svdd_loss` ✓.

## Findings for model 6

### Contract — PASS (verified live)

The adapter (`platform/wtpm_platform/adapters/anomaly.py`, `m13-deep-svdd`) uses the repo's API exactly as defined:

- `DeepSVDDNetwork(input_dim=X.shape[1], rep_dim=16)` — signature `(input_dim=64, rep_dim=32)` accepts both.
- `init_center(model, [(Xs,)])` — the adapter's 1-tuple batches match `for x, in loader`.
- `svdd_loss(model(Xs), model.c)` — centre is set by `init_center` before first use.

**Live replay of the adapter fit flow (torch 2.14, this session):** build → `init_center` → 60× `svdd_loss`/Adam — passed; centre shape `(16,)`, loss 0.1488 → 0.0038.

**Parameter counts (live torch):** default `(64, 32)` = **7,296**; platform `(38, 16)` = **5,104**.

### Defect — README is still the untouched scaffold

The current README contradicts `model.py`:

| README claim | Reality |
|---|---|
| `model.py  # Core neural network architecture / training script` | No training script — definitions only (`DeepSVDDNetwork`, `init_center`, `svdd_loss`); no `main` block |
| "Run the model: `python model.py`" | The command imports the definitions and exits; there is no dataset, loop, or output |
| (no model documentation) | No architecture, API, parameter count, platform-contract or limits section — every other audited sibling documents these |

`requirements.txt` lists `numpy` and `scikit-learn`, but `model.py` imports only `torch` (`torch.nn`) — harmless over-listing, documented in the fixed README rather than changed. Landing pages (`index.html`, `docs/index.html`) make no numeric claims; no change needed.

### Fix — replace README with the sibling-standard version

Paste-ready corrected README: [`docs/model-6-deep-svdd-readme.fixed.md`](model-6-deep-svdd-readme.fixed.md) in this commit (drop-in replacement for `README.md` in `wt-pm-deep-svdd-boundary`). It documents the real API, both parameter counts (7,296 default / 5,104 platform), the platform contract (`m13-deep-svdd`, fallback `m14-isolation-forest`), and honest limits. Docs-only change; `model.py` untouched.

## Lock impact

Docs-only change: `model.py` is unmodified, so its lock checksum stays valid; the repo's `revision` pin advances when the README fix merges (re-run the re-pin commands from [audit-models-1-4-fixes.md](audit-models-1-4-fixes.md), Fix 3, alongside the convlstm/aerozip merges).

## Status

| item | state |
|---|---|
| Adapter contract (`DeepSVDDNetwork` / `init_center` / `svdd_loss`) | verified live — no code change needed |
| README refresh | paste-ready file committed here; issue filed on the repo; push requires that repo's session |
