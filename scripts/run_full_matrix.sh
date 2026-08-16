#!/usr/bin/env bash
set -euo pipefail
for regime in attack_heavy diverse_attack authorization_balanced; do
  bash scripts/train_full.sh "$regime"
done
