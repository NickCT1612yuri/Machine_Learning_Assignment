# Extracted features

This directory holds per-dataset embedding caches written by the pipeline:

```
features/<dataset>/<embedding>_<preprocess_tag>/
  X_train.npy | X_train_sparse.npz
  X_test.npy  | X_test_sparse.npz
  X_train_pca30.npy, X_test_pca30.npy
  y_train.npy, y_test.npy
  features.h5, meta.json
```

The binary caches are **not** committed to GitHub (they exceed typical repository size limits). Regenerate them locally or in Colab:

```bash
python main.py --dataset all
```

or **Runtime → Run all** in `notebooks/text_classification_colab.ipynb`.

For the course zip submission, attach a packed `features/` archive alongside the repository if the instructor requires the `.npy` / `.h5` files on disk.
