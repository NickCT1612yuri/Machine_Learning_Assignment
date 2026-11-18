"""
Run the Banking77 text-classification assignment from the terminal.

Examples
--------
python main.py --quick                 # smoke test on a small subset
python main.py                         # full Banking77 benchmark
python main.py --skip-bilstm           # classical pipeline only
python main.py --skip-ablation
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Configurable Banking77 text-classification benchmark")
    parser.add_argument(
        "--dataset",
        default="banking77",
        choices=["banking77"],
        help="Dataset to run (assignment scope is Banking77 only)",
    )
    parser.add_argument("--quick", action="store_true", help="small subsets for a fast end-to-end check")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-bilstm", action="store_true")
    args = parser.parse_args()

    if args.quick:
        os.environ["TC_QUICK"] = "1"
    import matplotlib

    matplotlib.use("Agg")

    from modules.config import DATASET_ORDER
    from modules.evaluation import build_master_table, run_all_eda, run_dataset_benchmark_suite
    from modules.utils import seed_everything

    seed_everything()
    names = DATASET_ORDER if args.dataset == "banking77" else [args.dataset]
    print(run_all_eda(names).T.to_string())
    for name in names:
        print("\n" + "=" * 70 + f"\n{name}\n" + "=" * 70)
        results, _ = run_dataset_benchmark_suite(
            name,
            run_ablation=not args.skip_ablation,
            run_bilstm_model=not args.skip_bilstm,
        )
        cols = ["method", "accuracy", "f1", "brier_score", "ece", "train_time_sec", "infer_time_sec"]
        print(results[cols].round(4).to_string(index=False))
    master = build_master_table(names)
    print(f"\nmaster table: {len(master)} rows -> reports/master_comparison.csv")


if __name__ == "__main__":
    main()
