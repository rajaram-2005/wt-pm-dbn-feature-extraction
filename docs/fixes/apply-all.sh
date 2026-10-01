#!/usr/bin/env bash
# One-shot applier for the verified audit fixes (models 1-4 audit + model 6).
#
# WHAT THIS DOES
#   1. wt-pm-convlstm-wear-prognostics : add build_convlstm alias   (patch 0001)
#   2. wt-pm-aerozip-autoencoder-compressor : README param count    (patch 0002)
#   3. wt-pm-deep-svdd-boundary        : README refresh             (patch 0003)
# Each fix is applied on a branch, pushed, and opened as a PR.
# Afterwards, run 0004-repin-lock.sh (after merging the PRs) to re-pin the lock.
#
# REQUIREMENTS
#   - gh authenticated with WRITE access to the three repos above
#     (the audit session's installation was scoped to wt-pm-dbn-feature-extraction
#      only, which is why these were never pushed from there).
#   - git, python3. No ML frameworks needed: all three patches are verified
#     text changes; runtime verification commands are printed at the end.
#
# All patches were generated against main HEADs of 2026-10-01 and dry-run
# applied cleanly; the underlying fixes were verified live (TF 2.21 / torch 2.14).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

apply_fix() {
  local repo="$1" patch="$2" branch="$3" title="$4" body="$5"
  echo "=== $repo ==="
  git clone --quiet "https://github.com/rajaram-2005/$repo.git" "$WORK/$repo"
  cd "$WORK/$repo"
  git checkout -q -b "$branch"
  git apply --check "$HERE/$patch"
  git apply "$HERE/$patch"
  git add -A
  git commit -q -m "$title" -m "$body"
  git push -q origin "$branch"
  gh pr create --title "$title" --body "$body"
  cd "$WORK"
}

apply_fix wt-pm-convlstm-wear-prognostics 0001-convlstm-build-alias.patch \
  fix/build-convlstm-alias \
  "Add build_convlstm alias expected by the platform adapter" \
"The wt-pm adapter m02-convlstm-wear calls mod.build_convlstm(input_shape=(S, GRID, GRID, 1)) but model.py only defined build_model_4, so the adapter crashed at fit time. This aliases the factory under the expected name; architecture unchanged. Verified live (TF 2.21): alias resolves and build_convlstm() builds a 960,001-param model, matching the README's ~0.96M. See rajaram-2005/wt-pm-dbn-feature-extraction docs/audit-models-1-4-fixes.md (Fix 1)."

apply_fix wt-pm-aerozip-autoencoder-compressor 0002-aerozip-readme-param-count.patch \
  fix/readme-param-count \
  "Fix README param count for platform configuration (2,468 -> 2,794)" \
"The platform adapter instantiates AeroZipCompressor(input_dim=38, latent_dim=4): Linear(38,32)=1,248 + Linear(32,4)=132 + Linear(4,32)=160 + Linear(32,38)=1,254 = 2,794 parameters, not 2,468. The default (64, 8) count 4,744 is correct and unchanged. Verified live with torch. Docs-only change. See rajaram-2005/wt-pm-dbn-feature-extraction docs/audit-models-1-4-fixes.md (Fix 2)."

apply_fix wt-pm-deep-svdd-boundary 0003-deep-svdd-readme-refresh.patch \
  fix/readme-refresh \
  "Refresh scaffold README to match model.py (docs-only)" \
"The README was still the untouched scaffold: it called model.py a training script (there is none - definitions only) and said python model.py runs the model (it only defines the module). This replaces it with the sibling-standard README documenting the real API, parameter counts (7,296 default / 5,104 platform), the m13-deep-svdd contract and honest limits. Contract verified live: adapter fit flow replayed end-to-end (torch 2.14), loss 0.1488 -> 0.0038. See rajaram-2005/wt-pm-dbn-feature-extraction docs/audit-model-6-deep-svdd.md."

cat <<'DONE'

All three PRs opened. NEXT STEPS:
  1. Merge the three PRs.
  2. Clone wt-pm-lstm-scada-anomaly and run 0004-repin-lock.sh from this
     directory to re-pin deployment/models.lock.json (it validates via
     deployment/fetch_models.py and opens the lock PR).

Optional runtime verification (needs TF / torch):
  convlstm: python -c "from model import build_convlstm, build_model_4; assert build_convlstm is build_model_4; print(build_convlstm().count_params())"   # -> 960001
  aerozip:  python -c "from model import AeroZipCompressor; print(sum(p.numel() for p in AeroZipCompressor(38, 4).parameters()))"                          # -> 2794
DONE
