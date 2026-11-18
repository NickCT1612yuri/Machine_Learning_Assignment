# Extracted features

This directory holds Banking77 embedding caches written by the pipeline:

```
features/banking77/<embedding>_<preprocess_tag>/
  X_train.npy | X_train_sparse.npz
  X_test.npy  | X_test_sparse.npz
  X_train_pca30.npy, X_test_pca30.npy
  y_train.npy, y_test.npy
  features.h5, meta.json
  word2vec.kv.npz   # Word2Vec only
```

Default embeddings: `tfidf`, `word2vec`, `distilbert`. Binary caches are **not** committed to GitHub. Regenerate:

```bash
python main.py
```

or **Runtime → Run all** in `notebooks/text_classification_colab.ipynb`.

For the course zip, attach a packed `features/` archive if the instructor requires the `.npy` / `.h5` files on disk.
