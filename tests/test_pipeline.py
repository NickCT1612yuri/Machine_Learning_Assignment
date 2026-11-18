"""Fast checks of the properties the experiment relies on (no downloads). Run: python -m unittest -v"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from modules.data import Corpus, clean_text, load_corpus
from modules.evaluation import brier, ece
from modules.features import fit_tfidf, load_sparse, make_vectorizer, save_sparse
from modules.pipeline import select_tfidf


def toy_corpus(test_labels=(0, 1)):
    fit = ["red good", "red nice", "blue bad", "blue poor"] * 4
    return Corpus("toy", {"fit": fit, "val": ["red good", "blue bad"], "test": ["red nice", "blue poor"]},
                  {"fit": np.array([0, 0, 1, 1] * 4), "val": np.array([0, 1]), "test": np.array(test_labels)},
                  ["pos", "neg"], max_len=8)


class PipelineTests(unittest.TestCase):
    def test_clean_text_repairs_ag_news_entities(self):
        self.assertEqual(clean_text("It #39;s a  &lt;b&gt;deal&lt;/b&gt;\\now quot;hi quot;"), "It's a deal now \"hi \"")

    def test_vocabulary_comes_from_fit_split_only(self):
        corpus = toy_corpus()
        corpus.text["val"], corpus.text["test"] = ["valsecret valsecret"], ["testsecret testsecret"]
        vec = make_vectorizer("regex", False, 1)
        fit_tfidf(vec, corpus.text)
        self.assertNotIn("valsecret", vec.vocabulary_)
        self.assertNotIn("testsecret", vec.vocabulary_)

    def test_sparse_features_roundtrip_through_h5(self):
        corpus = toy_corpus()
        X = fit_tfidf(make_vectorizer("regex", False, 1), corpus.text)
        with tempfile.TemporaryDirectory() as d:
            save_sparse(Path(d) / "f.h5", X, corpus.y)
            restored = load_sparse(Path(d) / "f.h5")
        for s in X:
            np.testing.assert_array_equal(restored[s].toarray(), X[s].toarray())

    def test_selection_never_uses_test_labels(self):
        _, a = select_tfidf(toy_corpus((0, 1)), ["nb", "logreg"])
        _, b = select_tfidf(toy_corpus((1, 0)), ["nb", "logreg"])
        self.assertEqual(a, b)

    def test_test_duplicates_removed_from_development_only(self):
        class Split(dict):
            features = {"label": type("L", (), {"names": ["zero", "one"]})()}
            column_names = ["text", "label"]

            def __len__(self):
                return len(self["text"])

        train = Split(text=[f"text {i}" for i in range(40)], label=[i % 2 for i in range(40)])
        test = Split(text=["text 0", "unseen"], label=[0, 1])
        with patch("datasets.load_dataset", return_value={"train": train, "test": test}):
            corpus = load_corpus("rotten_tomatoes")
        self.assertEqual(corpus.text["test"], ["text 0", "unseen"])
        self.assertNotIn("text 0", corpus.text["fit"] + corpus.text["val"])

    def test_calibration_metrics_on_perfect_and_uniform_predictions(self):
        y = np.array([0, 1, 1])
        self.assertAlmostEqual(brier(y, np.eye(2)[y]), 0.0)
        self.assertAlmostEqual(ece(y, np.eye(2)[y]), 0.0)
        self.assertAlmostEqual(brier(y, np.full((3, 2), 0.5)), 0.5)


if __name__ == "__main__":
    unittest.main()
