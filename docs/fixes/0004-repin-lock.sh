#!/usr/bin/env bash
# Fix 4 — re-pin deployment/models.lock.json in wt-pm-lstm-scada-anomaly.
#
# Run this from a checkout of wt-pm-lstm-scada-anomaly where you can push,
# AFTER merging the convlstm alias + aerozip README + deep-svdd README fixes,
# so the new pins capture those commits:
#
#   git clone https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly.git
#   cd wt-pm-lstm-scada-anomaly && bash /path/to/0004-repin-lock.sh
#
# Requires: git, gh (authenticated), python3.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
test -f deployment/models.lock.json || { echo "Run from the wt-pm-lstm-scada-anomaly repo root"; exit 1; }

git checkout -b fix/models-lock-repin

python3 - <<'EOF'
import json, subprocess, base64, hashlib
lock = json.load(open('deployment/models.lock.json'))
changed = 0
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
    changed += 1
    print('re-pinned', name)
json.dump(lock, open('deployment/models.lock.json', 'w'), indent=2)
open('deployment/models.lock.json', 'a').write('\n')
print(f'{changed} of {len(lock["models"])} entries re-pinned')
EOF

# Validate end-to-end: clones every repo at its pinned revision and checks
# sha256(model.py). Must finish without a checksum mismatch.
python3 deployment/fetch_models.py --dest "$(mktemp -d)"

git add deployment/models.lock.json
git commit -m "Re-pin models.lock.json to current main HEADs (validated by fetch_models.py)"
git push -u origin fix/models-lock-repin
gh pr create --title "Re-pin deployment/models.lock.json (validated)" \
  --body "Re-pins drifted model repos to their current main HEADs; model_sha256 is sha256(model.py) at the pinned revision. Validated end-to-end: deployment/fetch_models.py cloned all repos at the pinned revisions and verified every checksum. See docs/audit-models-1-4-fixes.md (Fix 3) in rajaram-2005/wt-pm-dbn-feature-extraction for context."
echo "Done — review and merge the PR."
