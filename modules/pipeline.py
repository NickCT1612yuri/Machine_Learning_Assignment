"""End-to-end experiment for one corpus.

Order: load + clean + split -> EDA -> TF-IDF (traditional) -> BGE-small (modern features)
-> TextCNN (deep learning) -> metrics and error analysis.
Configurations are chosen on validation macro-F1; each chosen model is scored once on test.
"""
import itertools
import json
import platform
import time
from importlib.metadata import version

import numpy as np
import pandas as pd
import torch

from .config import BGE_C_GRID, BGE_MODEL, CLASSIFIER_GRID, DATASETS, ROOT, SEED, TEXTCNN, TFIDF_GRID
from .data import has_negation, load_corpus, run_eda
from .evaluation import bootstrap_f1, per_class_f1, scores, top_confusions
from .features import device, encode_bge, fit_tfidf, load_dense, load_sparse, make_vectorizer, save_dense, save_sparse
from .models import make_classifier, predict_proba, seed_everything, train_textcnn

FAMILIES = ("tfidf", "bge", "textcnn")


def _fit_and_score(name, param, X, corpus, scale=False):
    model = make_classifier(name, param, scale).fit(X["fit"], corpus.y["fit"])
    proba = predict_proba(model, X["val"], len(corpus.labels))
    return model, scores(corpus.y["val"], model.predict(X["val"]), proba)


def select_tfidf(corpus, classifiers):
    """Grid over tokenizer x stop words x n-grams x classifier x hyper-parameter (validation only)."""
    rows = []
    for tok, stop, ngram in itertools.product(*TFIDF_GRID.values()):
        X = fit_tfidf(make_vectorizer(tok, stop, ngram), corpus.text)
        for clf in classifiers:
            for param in CLASSIFIER_GRID[clf]:
                _, val = _fit_and_score(clf, param, X, corpus)
                rows.append({"family": "TF-IDF", "tokenizer": tok, "stop_words": stop, "max_ngram": ngram,
                             "model": clf, "param": param, **val})
    return rows, max(rows, key=lambda r: r["f1_macro"])


def run_tfidf(corpus, feature_dir, classifiers):
    rows, best = select_tfidf(corpus, classifiers)
    vectorizer = make_vectorizer(best["tokenizer"], best["stop_words"], best["max_ngram"])
    save_sparse(feature_dir / "tfidf.h5", fit_tfidf(vectorizer, corpus.text), corpus.y)
    X = load_sparse(feature_dir / "tfidf.h5")  # the classifier reads the stored features
    model = make_classifier(best["model"], best["param"]).fit(X["fit"], corpus.y["fit"])
    config = {k: best[k] for k in ("tokenizer", "stop_words", "max_ngram", "model", "param")}
    config["vocabulary"] = len(vectorizer.vocabulary_)
    return rows, config, model.predict(X["test"]), predict_proba(model, X["test"], len(corpus.labels))


def run_bge(corpus, feature_dir, padding):
    save_dense(feature_dir, encode_bge(corpus.text, corpus.max_len, padding))
    X = load_dense(feature_dir)
    rows, best = [], None
    for c in BGE_C_GRID:
        model, val = _fit_and_score("logreg", c, X, corpus, scale=True)
        rows.append({"family": "BGE-small", "model": "logreg", "param": c, **val})
        if best is None or val["f1_macro"] > best[1]["f1_macro"]:
            best = (model, val, c)
    model, _, c = best
    config = {"model": "logreg", "param": c, "dim": int(X["fit"].shape[1]), "padding": padding}
    return rows, config, model.predict(X["test"]), predict_proba(model, X["test"], len(corpus.labels))


def run_dataset(name, out_root=ROOT, quick=False, families=FAMILIES, classifiers=("nb", "logreg", "svm"),
                padding="dynamic", batch_size=32):
    seed_everything()
    results_dir = out_root / "reports" / "results" / name
    feature_dir = out_root / "features" / name
    results_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()

    def log(msg):
        print(f"[{name} {time.time() - start:6.0f}s] {msg}", flush=True)

    corpus = load_corpus(name, quick)
    log("fit/val/test = " + "/".join(str(len(corpus.y[s])) for s in corpus.y))
    eda = run_eda(corpus, results_dir)
    for s in corpus.y:
        np.save(feature_dir / f"y_{s}.npy", corpus.y[s])

    validation, configs, preds, probas = [], {}, {}, {}
    if "tfidf" in families:
        log("TF-IDF grid search")
        rows, configs["TF-IDF"], preds["TF-IDF"], probas["TF-IDF"] = run_tfidf(corpus, feature_dir, classifiers)
        validation += rows
    if "bge" in families:
        log("BGE-small embeddings")
        rows, configs["BGE-small"], preds["BGE-small"], probas["BGE-small"] = run_bge(corpus, feature_dir, padding)
        validation += rows
    if "textcnn" in families:
        log("TextCNN training")
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        pred, proba, history, info = train_textcnn(corpus, batch_size, 2 if quick else None, dev)
        preds["TextCNN"], probas["TextCNN"], configs["TextCNN"] = pred, proba, info
        pd.DataFrame(history).to_csv(results_dir / "textcnn_history.csv", index=False)

    log("evaluation")
    write_results(corpus, results_dir, validation, configs, preds, probas)
    meta = {"dataset": name, "source": DATASETS[name]["hf_id"], "labels": corpus.labels, "seed": SEED, "quick": quick,
            "max_len": corpus.max_len, "batch_size": batch_size, "bge_model": BGE_MODEL, "textcnn": TEXTCNN,
            "bge_device": str(device()), "environment": _environment()}
    (results_dir / "run.json").write_text(json.dumps(meta, indent=2))
    (results_dir / "eda.json").write_text(json.dumps(eda, indent=2))
    return pd.read_csv(results_dir / "test_results.csv")


def write_results(corpus, out, validation, configs, preds, probas):
    y, labels, texts = corpus.y["test"], corpus.labels, corpus.text["test"]
    pd.DataFrame(validation).to_csv(out / "validation.csv", index=False)

    draws = bootstrap_f1(y, preds)
    rows = [{"family": f, "config": json.dumps(configs[f]), **scores(y, preds[f], probas[f]),
             "f1_ci_low": np.percentile(draws[f], 2.5), "f1_ci_high": np.percentile(draws[f], 97.5)}
            for f in preds]
    results = pd.DataFrame(rows)
    results.to_csv(out / "test_results.csv", index=False)
    print(results[["family", "accuracy", "f1_macro", "f1_ci_low", "f1_ci_high"]].round(4).to_string(index=False))

    # Paired bootstrap: how often the best family beats each other family on the same resample.
    best = results.loc[results.f1_macro.idxmax(), "family"]
    pd.DataFrame([{"best": best, "other": f, "mean_gap": (draws[best] - draws[f]).mean(),
                   "gap_ci_low": np.percentile(draws[best] - draws[f], 2.5),
                   "gap_ci_high": np.percentile(draws[best] - draws[f], 97.5),
                   "p_best_wins": (draws[best] > draws[f]).mean()} for f in preds if f != best]
                 ).to_csv(out / "paired_bootstrap.csv", index=False)

    negated = np.array([has_negation(t) for t in texts])
    pd.DataFrame([{"family": f, "n_negation": int(negated.sum()), "n_other": int((~negated).sum()),
                   "acc_negation": (p == y)[negated].mean(), "acc_other": (p == y)[~negated].mean()}
                  for f, p in preds.items()]).assign(gap=lambda d: d.acc_negation - d.acc_other
                                                     ).to_csv(out / "negation.csv", index=False)

    per_class_f1(y, preds, labels).to_csv(out / "per_class_f1.csv")
    pd.concat([top_confusions(y, p, labels).assign(family=f) for f, p in preds.items()]
              ).to_csv(out / "top_confusions.csv", index=False)
    pd.DataFrame({"text": texts, "true": [labels[i] for i in y],
                  **{f: [labels[i] for i in p] for f, p in preds.items()}}).to_csv(out / "predictions.csv", index=False)


def _environment():
    packages = ("numpy", "pandas", "scikit-learn", "scipy", "torch", "transformers", "datasets", "h5py")
    return {"python": platform.python_version(), **{p: version(p) for p in packages}}
