"""Command-line entry point: python main.py [--datasets ...] [--quick]."""
import argparse
from pathlib import Path

import pandas as pd

from modules import DATASETS, FAMILIES, run_dataset
from modules.config import ROOT


def main():
    p = argparse.ArgumentParser(description="TF-IDF vs BGE-small vs TextCNN text classification")
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    p.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    p.add_argument("--classifiers", nargs="+", choices=["nb", "logreg", "svm"], default=["nb", "logreg", "svm"])
    p.add_argument("--padding", choices=["dynamic", "fixed"], default="dynamic", help="BGE batch padding")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--quick", action="store_true", help="small subsets, 2 epochs; writes to .smoke/")
    p.add_argument("--out", type=Path, help="output root (default: repository, or .smoke/ with --quick)")
    a = p.parse_args()

    out = a.out or (ROOT / ".smoke" if a.quick else ROOT)
    tables = [run_dataset(n, out, a.quick, a.families, a.classifiers, a.padding, a.batch_size).assign(dataset=n)
              for n in a.datasets]
    summary = pd.concat(tables)
    summary.to_csv(out / "reports" / "results" / "summary.csv", index=False)
    print(f"\nDone. Results: {out / 'reports' / 'results'}  Features: {out / 'features'}")


if __name__ == "__main__":
    main()
