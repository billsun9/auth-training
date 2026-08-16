#!/usr/bin/env bash
set -euo pipefail
for regime in capability_only attack_heavy diverse_attack authorization_balanced; do
  bash scripts/train_full.sh "$regime"
done
