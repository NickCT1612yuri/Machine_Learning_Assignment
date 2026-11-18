"""Build LaTeX tables and figures for the report from reports/results/ (run by `make`)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
OUT = HERE / "generated"
NAMES = {"ag_news": "AG News", "rotten_tomatoes": "Rotten Tomatoes", "banking77": "Banking77"}
FAMILIES = ["TF-IDF", "BGE-small", "TextCNN"]


def pct(x):
    return "--" if pd.isna(x) else f"{100 * x:.2f}"


def num(x, d=3):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def table(spec, header, rows, caption, label, size="\\small"):
    body = "\n".join(" & ".join(map(str, r)) + r" \\" if not isinstance(r, str) else r for r in rows)
    return (f"\\begin{{table}}[H]\n\\centering\n{size}\n\\caption{{{caption}}}\\label{{{label}}}\n"
            f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n"
            "\\end{tabular}\n\\end{table}\n")


def read(name, file):
    return pd.read_csv(RESULTS / name / file)


def eda_table():
    rows = []
    for n, label in NAMES.items():
        e = json.loads((RESULTS / n / "eda.json").read_text())
        rows.append([label, f"{e['n_fit']:,} / {e['n_val']:,} / {e['n_test']:,}", e["classes"],
                     f"{e['min_class']}--{e['max_class']}", f"{e['length_mean']:.1f}", f"{e['length_p95']:.0f}",
                     f"{e['vocabulary']:,}", f"{100 * e['negation_share']:.1f}",
                     e["train_duplicates"], e["removed_from_fit"] + e["removed_from_val"]])
    return table("@{}lrrrrrrrrr@{}",
                 "Corpus & Fit / Val / Test & $K$ & Class size & Mean len & P95 & Vocab & Neg.\\,\\% & Dup. & Leak rm.",
                 rows, "Fit-split statistics after cleaning (lengths in regex word tokens; class size = min--max "
                       "examples per class; Neg.\\,\\% = texts with a negation cue; Dup. = duplicate texts in the "
                       "original training split; Leak rm. = development texts removed because they also occur in a "
                       "later split).", "tab:eda", "\\footnotesize")


def main_table():
    rows = []
    for i, (n, label) in enumerate(NAMES.items()):
        r = read(n, "test_results.csv")
        best = r.f1_macro.max()
        if i:
            rows.append("\\midrule")
        for _, x in r.iterrows():
            f1 = pct(x.f1_macro)
            f1 = f"\\textbf{{{f1}}}" if x.f1_macro == best else f1
            rows.append([label if _ == 0 else "", x.family, pct(x.accuracy), pct(x.precision_macro),
                         pct(x.recall_macro), f1, f"[{pct(x.f1_ci_low)}, {pct(x.f1_ci_high)}]",
                         num(x.brier), num(x.ece)])
    return table("@{}llrrrrcrr@{}", "Corpus & Family & Acc & P$_{\\text{macro}}$ & R$_{\\text{macro}}$ & "
                 "F1$_{\\text{macro}}$ & 95\\% CI (F1) & Brier & ECE", rows,
                 "Held-out test results (\\%, except Brier and ECE; lower is better for both). "
                 "Bold: best macro-F1 per corpus. CI: percentile bootstrap, 1{,}000 resamples. "
                 "LinearSVC gives no probabilities, so Brier/ECE are undefined for it.", "tab:main", "\\footnotesize")


def selected_table():
    rows = []
    for n, label in NAMES.items():
        r = read(n, "test_results.csv").set_index("family")
        t = json.loads(r.loc["TF-IDF", "config"])
        b = json.loads(r.loc["BGE-small", "config"])
        c = json.loads(r.loc["TextCNN", "config"])
        model = {"nb": "NB ($\\alpha$={p})", "logreg": "LogReg (C={p})", "svm": "LinearSVC (C={p})"}[t["model"]]
        rows.append([label, f"{t['tokenizer']}, stop words {'removed' if t['stop_words'] else 'kept'}, "
                            f"1--{t['max_ngram']}-grams, {model.format(p=t['param'])}",
                     f"C={b['param']}", f"epoch {c['best_epoch']}, vocab {c['vocab_size']:,}"])
    return table("@{}lP{7.4cm}ll@{}", "Corpus & TF-IDF (best of 112) & BGE (best of 7) & TextCNN", rows,
                 "Configurations selected by validation macro-F1.", "tab:selected", "\\footnotesize")


def paired_table():
    rows = []
    for n, label in NAMES.items():
        for _, x in read(n, "paired_bootstrap.csv").iterrows():
            rows.append([label, f"{x.best} $-$ {x.other}", f"{100 * x.mean_gap:+.2f}",
                         f"[{100 * x.gap_ci_low:+.2f}, {100 * x.gap_ci_high:+.2f}]", f"{x.p_best_wins:.3f}"])
    return table("@{}llrcr@{}", "Corpus & Comparison & $\\Delta$F1 (pp) & 95\\% CI & P(best wins)", rows,
                 "Paired bootstrap of the macro-F1 gap between the best family and each other family "
                 "(same 1{,}000 test resamples for both models).", "tab:paired")


def validation_table():
    """Best validation F1 per classifier, and the effect of each preprocessing switch (best-of contrast)."""
    rows = []
    for n, label in NAMES.items():
        v = read(n, "validation.csv")
        t = v[v.family == "TF-IDF"]
        best = {m: t[t.model == m].f1_macro.max() for m in ("nb", "logreg", "svm")}

        def delta(col, a, b):
            return 100 * (t[t[col] == b].f1_macro.max() - t[t[col] == a].f1_macro.max())

        top = max(best.values())
        cells = [f"\\textbf{{{pct(x)}}}" if x == top else pct(x) for x in best.values()]
        rows.append([label, *cells, f"{delta('stop_words', False, True):+.2f}",
                     f"{delta('max_ngram', 1, 2):+.2f}", f"{delta('tokenizer', 'regex', 'whitespace'):+.2f}",
                     pct(v[v.family == "BGE-small"].f1_macro.max())])
    return table("@{}lrrrrrrr@{}", "Corpus & NB & LogReg & SVM & Stop words & Bigrams & Whitespace & BGE",
                 rows, "Validation macro-F1 (\\%). Left: best TF-IDF configuration per classifier. "
                       "Middle: change in the best achievable score (pp) when a switch is turned on "
                       "(stop-word removal; adding bigrams; whitespace instead of regex tokenizer), "
                       "maximising over all other settings. Right: best BGE-small score.", "tab:validation",
                 "\\footnotesize")


def negation_table():
    rows = []
    for n, label in NAMES.items():
        for i, x in read(n, "negation.csv").iterrows():
            rows.append([label if i == 0 else "", x.family, f"{int(x.n_negation)} / {int(x.n_other)}",
                         pct(x.acc_negation), pct(x.acc_other), f"{100 * x.gap:+.2f}"])
    return table("@{}llrrrr@{}", "Corpus & Family & $n$ neg./other & Acc neg. & Acc other & Gap (pp)", rows,
                 "Test accuracy on texts with and without a negation cue (\\%).", "tab:negation")


def confusion_table():
    c = read("banking77", "top_confusions.csv")
    c = c[c.family == "BGE-small"].head(8)
    esc = lambda s: s.replace("_", "\\_")
    rows = [[f"\\texttt{{{esc(x.true)}}}", f"\\texttt{{{esc(x.predicted)}}}", x["count"]] for _, x in c.iterrows()]
    return table("@{}llr@{}", "True intent & Predicted intent & Count", rows,
                 "Most frequent Banking77 test confusions of the best model (BGE-small).", "tab:confusions",
                 "\\footnotesize")


def worst_classes_table():
    p = read("banking77", "per_class_f1.csv").sort_values("BGE-small").head(8)
    esc = lambda s: s.replace("_", "\\_")
    rows = [[f"\\texttt{{{esc(x['class'])}}}", *(pct(x[f]) for f in FAMILIES)] for _, x in p.iterrows()]
    return table("@{}lrrr@{}", "Intent (lowest BGE F1) & TF-IDF & BGE-small & TextCNN", rows,
                 "Banking77 intents with the lowest test F1 for BGE-small (\\%).", "tab:worst", "\\footnotesize")


def textcnn_figure():
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    for n, label in NAMES.items():
        h = read(n, "textcnn_history.csv")
        ax[0].plot(h.epoch, h.train_loss, marker="o", ms=3, label=label)
        ax[1].plot(h.epoch, 100 * h.f1_macro, marker="o", ms=3, label=label)
    ax[0].set(xlabel="epoch", ylabel="training loss", title="Training loss")
    ax[1].set(xlabel="epoch", ylabel="validation macro-F1 (%)", title="Validation macro-F1")
    ax[1].legend(frameon=False)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "textcnn_curves.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, fn in [("eda", eda_table), ("main", main_table), ("selected", selected_table),
                     ("paired", paired_table), ("validation", validation_table), ("negation", negation_table),
                     ("confusions", confusion_table), ("worst", worst_classes_table)]:
        (OUT / f"{name}.tex").write_text(fn())
    textcnn_figure()
    print("tables and figures written to", OUT)
