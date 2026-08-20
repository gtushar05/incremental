#!/bin/bash
# Deploy the static cockpit to the HF Space from a fresh single-commit
# branch. Keeps main's history untouched (HF requires binaries via LFS,
# and we will not rewrite the pre-registered provenance on GitHub for a
# hosting requirement).
set -euo pipefail
cd "$(dirname "$0")/.."

[ -z "$(git status --porcelain)" ] || { echo "working tree not clean — commit first"; exit 1; }
git rev-parse --verify main >/dev/null

BRANCH=space-deploy
git branch -D $BRANCH 2>/dev/null || true
git checkout --orphan $BRANCH -q
git rm -r --cached . -q

git lfs install --local >/dev/null
git lfs track "reports/*.png" >/dev/null
git add .gitattributes README.md index.html app.py requirements.txt app/ reports/
git commit -qm "Space deploy: static cockpit + precomputed artifacts ($(date +%Y-%m-%d))"
git push space $BRANCH:main --force

git checkout -f main -q
git branch -D $BRANCH -q
rm -f .gitattributes
echo "deployed -> https://huggingface.co/spaces/Gtushar-05/incremental-cockpit"
