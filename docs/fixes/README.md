# Verified fix bundle — models 1–4 audit + model 6

Everything here was generated and verified in the audit sessions (see
[audit-models-1-4-fixes.md](../audit-models-1-4-fixes.md) and
[audit-model-6-deep-svdd.md](../audit-model-6-deep-svdd.md)). The audit
session's GitHub installation is scoped to `wt-pm-dbn-feature-extraction`
only, so these fixes could not be pushed to their target repos from there.
This bundle lets anyone (or any session) with write access apply them in one shot.

## Contents

| file | target repo | fix | verified |
|---|---|---|---|
| `0001-convlstm-build-alias.patch` | wt-pm-convlstm-wear-prognostics | add `build_convlstm` alias (adapter m02 crashes without it) | live, TF 2.21 — 960,001 params |
| `0002-aerozip-readme-param-count.patch` | wt-pm-aerozip-autoencoder-compressor | README platform-config count 2,468 → 2,794 | live, torch 2.14 |
| `0003-deep-svdd-readme-refresh.patch` | wt-pm-deep-svdd-boundary | replace stale scaffold README (docs-only) | contract replayed live, torch 2.14 |
| `0004-repin-lock.sh` | wt-pm-lstm-scada-anomaly | re-pin `deployment/models.lock.json` + validate with `fetch_models.py` | re-pin validated 24/24 for the 2026-09-30 HEADs (see `../models.lock.repin.json`) |
| `apply-all.sh` | — | applies patches 1–3: clone → branch → apply → push → PR | dry-run applied cleanly against 2026-10-01 main HEADs |

## How to apply

```bash
# requires: gh authenticated with write access to the three target repos
bash apply-all.sh          # opens 3 PRs
# merge the 3 PRs, then:
git clone https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly.git
cd wt-pm-lstm-scada-anomaly && bash <path-to>/0004-repin-lock.sh   # opens the lock PR
```

Patches were generated with `diff -u` against each repo's `main` HEAD of
2026-10-01 (convlstm `21aa8e9`, aerozip `a615786`, deep-svdd `a2aa793`) and
apply cleanly with `git apply` / `patch -p1`. If a repo's `main` has moved,
`git apply --check` will tell you before anything is committed.

## Same fixes, other formats

- Issue on each target repo carries the same patches inline:
  convlstm [#2](https://github.com/rajaram-2005/wt-pm-convlstm-wear-prognostics/issues/2),
  aerozip [#2](https://github.com/rajaram-2005/wt-pm-aerozip-autoencoder-compressor/issues/2),
  deep-svdd [#1](https://github.com/rajaram-2005/wt-pm-deep-svdd-boundary/issues/1),
  platform lock [#5](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/issues/5).
- `../models.lock.repin.json` is the lock as validated for the 2026-09-30
  HEADs (before fixes 1–3 merge); regenerate with `0004-repin-lock.sh` after.
