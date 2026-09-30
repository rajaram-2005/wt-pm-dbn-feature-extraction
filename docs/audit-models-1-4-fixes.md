# Audit — Models 1–4: findings and ready-to-apply fixes

**Date:** 2026-09-30 · **Scope:** first four model repos alphabetically (`wt-pm-1d-cnn-bearing-vibration`, `wt-pm-aerozip-autoencoder-compressor`, `wt-pm-contrastive-ssl-vibration`, `wt-pm-convlstm-wear-prognostics`) plus the platform lock (`wt-pm-lstm-scada-anomaly` `deployment/models.lock.json`).
**Method:** every adapter symbol call was checked against the repo's `model.py` at its current `main` HEAD; every README number was recomputed from the architecture; the lock was re-pinned and validated end-to-end with `fetch_models.py`.

> **Access note:** this session's GitHub installation is scoped to
> `wt-pm-dbn-feature-extraction` only (verified via `GET /installation/repositories`),
> so the fixes below are **prepared and fully verified here** but must be pushed from
> the per-repo sessions (or by a maintainer). Patches are exact and paste-ready.

---

## Fix 1 — `build_convlstm` alias in `wt-pm-convlstm-wear-prognostics`

**Defect (runtime crash).** The platform adapter `m02-convlstm-wear` calls:

```python
# platform/wtpm_platform/adapters/prognostics.py:221
model = mod.build_convlstm(input_shape=(S, self.GRID, self.GRID, 1))
```

but `model.py` only defines `build_model_4` — the adapter raises `AttributeError` at `fit()`.

**Patch** (append after the `build_model_4` definition, ~line 137 of `model.py`):

```python


# Platform alias: the wt-pm adapter (m02-convlstm-wear) imports the factory by this name.
build_convlstm = build_model_4
```

**Verification one-liner** (needs TensorFlow; run from the repo root):

```bash
python -c "from model import build_convlstm, build_model_4; assert build_convlstm is build_model_4; print(build_convlstm().count_params())"
# -> 960001   (matches README's "~0.96M parameters")
```

*Executed this session with TF 2.21: passed.*

---

## Fix 2 — AeroZip README param-count correction (`wt-pm-aerozip-autoencoder-compressor`)

**Defect (docs).** README "Size" line claims **2,468** parameters for the platform configuration. The adapter (`classification.py`, `m23-aerozip`) instantiates `AeroZipCompressor(input_dim=38, latent_dim=4)` (`latent = max(cols // 8, 4)`), which is:

| layer | params |
|---|---|
| `Linear(38, 32)` | 1,248 |
| `Linear(32, 4)` | 132 |
| `Linear(4, 32)` | 160 |
| `Linear(32, 38)` | 1,254 |
| **total** | **2,794** |

**Patch** (single line in `README.md`, ~line 26):

```diff
-- **Size:** 4,744 parameters for the default `(input_dim=64, latent_dim=8)` configuration (2,468 parameters when the platform instantiates it on its 38-column feature table with `latent_dim=4`).
+- **Size:** 4,744 parameters for the default `(input_dim=64, latent_dim=8)` configuration (2,794 parameters when the platform instantiates it on its 38-column feature table with `latent_dim=4`).
```

The default-config figure (4,744) is correct and unchanged.

**Verification** (torch):

```bash
python -c "from model import AeroZipCompressor; print(sum(p.numel() for p in AeroZipCompressor(38, 4).parameters()), sum(p.numel() for p in AeroZipCompressor(64, 8).parameters()))"
# -> 2794 4744
```

*Executed this session with torch 2.14: passed.* Docs-only change; no other occurrence of the wrong figure exists in the repo (checked `index.html`, `docs/index.html`).

---

## Fix 3 — `models.lock.json` re-pin (`wt-pm-lstm-scada-anomaly`)

**Defect (stale pins).** 5 of 24 lock entries drift from their repo's `main` HEAD; the other 19 are in sync. Most notably the dbn entry still pins `67b3ffa` — the *initial scaffold* — missing the Model 5 work merged in PR #1.

| repository | old pin | new pin (`main` HEAD, 2026-09-30) |
|---|---|---|
| wt-pm-1d-cnn-bearing-vibration | `de745130a4…` | `04a846f158755308b8c88d4600707972ad70c1a3` |
| wt-pm-aerozip-autoencoder-compressor | `4e52239afc…` | `a61578612e01970ffc6b08273e55d6b330938920` |
| wt-pm-contrastive-ssl-vibration | `8cbab8b524…` | `0626f1476ec5262ca6ffc3a1e5e6a3fd0fd70934` |
| wt-pm-convlstm-wear-prognostics | `ca64904fbf…` | `21aa8e9b3df4f5bbfaf8a4f73f95e8c92f926ae6` |
| wt-pm-dbn-feature-extraction | `67b3ffae88…` | `b69e2e7b7f62f0d7214b98901e4d0acfd4674020` |

**Verified re-pinned lock file:** [`docs/models.lock.repin.json`](models.lock.repin.json) in this commit. `model_sha256` is `sha256(model.py bytes at the pinned revision)` — the exact check `fetch_models.py` enforces.

**Validation performed this session:**

```bash
python deployment/fetch_models.py --lock docs/models.lock.repin.json --dest "$(mktemp -d)"
# -> all 24 repos cloned at pinned revisions, all 24 model.py checksums verified — PASSED
```

**Re-pin regeneration commands** (re-run after Fixes 1–2 merge, since their merges move those repos' HEADs):

```bash
cd deployment   # inside wt-pm-lstm-scada-anomaly
python3 - <<'EOF'
import json, subprocess, base64, hashlib
lock = json.load(open('models.lock.json'))
for row in lock['models']:
    name = row['repository']
    head = subprocess.run(['git', 'ls-remote',
                           f'https://github.com/rajaram-2005/{name}.git',
                           'refs/heads/main'],
                          capture_output=True, text=True, check=True).stdout.split()[0]
    if head == row['revision']:
        continue
    blob = subprocess.run(['gh', 'api',
                           f'repos/rajaram-2005/{name}/contents/model.py?ref={head}',
                           '--jq', '.content'],
                          capture_output=True, text=True, check=True).stdout
    row['revision'] = head
    row['model_sha256'] = hashlib.sha256(base64.b64decode(blob)).hexdigest()
    print('re-pinned', name)
json.dump(lock, open('models.lock.json', 'w'), indent=2)
open('models.lock.json', 'a').write('\n')
EOF
python fetch_models.py --dest "$(mktemp -d)"   # must finish without a checksum mismatch
```

**Order of operations:** merge Fix 1 → merge Fix 2 → run the re-pin commands above → open the lock PR with the final file. (The attached `models.lock.repin.json` is already correct for every repo *except* that Fix 1/Fix 2 merges will advance the convlstm and aerozip pins again.)

---

## Cross-checks performed — no further defects

- Adapter symbols verified present at current `main` for every audited repo:
  `build_1d_cnn` (m01) ✓ · `AeroZipCompressor` (m23) ✓ · `ContrastiveEncoder` (m08) ✓ · `DBN` (m09) ✓ · `build_convlstm` (m02) ✗ — the only broken contract, fixed above.
- The 19 non-drifted lock entries match their repos' `main` HEADs exactly.
- Platform `docs/sibling-readmes/` does not repeat the wrong 2,468 figure.

## Status

| fix | target repo | state |
|---|---|---|
| 1 — alias | wt-pm-convlstm-wear-prognostics | patch verified (TF 2.21, 960,001 params); push requires that repo's session |
| 2 — README count | wt-pm-aerozip-autoencoder-compressor | patch verified (torch 2.14: 2,794); push requires that repo's session |
| 3 — lock re-pin | wt-pm-lstm-scada-anomaly | lock regenerated + `fetch_models.py` validation passed 24/24; re-run regeneration after Fixes 1–2 merge, then push |
