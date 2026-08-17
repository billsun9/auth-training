#!/usr/bin/env python
"""Create one clear cross-model comparison plot from completed experiment runs."""

import argparse

from auth_sft.plotting import plot_model_comparison


def main():
    parser = argparse.ArgumentParser(description="Compare final eval metrics across experiment runs")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(plot_model_comparison(args.runs_root, args.output))


if __name__ == "__main__":
    main()
