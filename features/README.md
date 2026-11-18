# Extracted features

Written by `python main.py` (or the Colab notebook); not tracked by git because of their size.

```
features/<corpus>/
  tfidf.h5                 selected TF-IDF configuration; groups fit/val/test with CSR data, indices, indptr, shape, y
  bge_fit.npy, bge_val.npy, bge_test.npy    frozen BGE-small embeddings (N x 384, float32)
  y_fit.npy, y_val.npy, y_test.npy          label ids
```

Corpora: `ag_news`, `rotten_tomatoes`, `banking77`. Attach this folder (or a download link) to the submission zip.
