#!/usr/bin/env bash
set -euo pipefail
for regime in attack_heavy authorization_balanced; do
  bash scripts/train_full.sh "$regime"
done
