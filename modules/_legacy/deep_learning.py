"""
Bonus deep-learning pipeline: Word2Vec + BiLSTM for sequence modelling.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from modules.classifiers import compute_metrics
from modules.config import BILSTM, DL_EMBEDDINGS, DL_MODELS, FEATURE_DIR, SEED, ExperimentConfig, PreprocessConfig
from modules.data_loader import quick_mode
from modules.feature_extraction import cache_dir_for, extract_and_save_features, train_or_load_word2vec
from modules.utils import get_device

UNPADDED_MAX_LEN = 10


def _resolve_max_len(lengths, prep: PreprocessConfig, cap: int) -> int:
    if not prep.padding:
        return UNPADDED_MAX_LEN
    p95 = int(np.ceil(np.percentile(lengths, 95))) if len(lengths) else 20
    return int(min(cap, max(20, p95)))


def _pad_ids(id_lists, max_len, pad_id=0):
    mat = np.full((len(id_lists), max_len), pad_id, dtype=np.int64)
    lengths = np.ones(len(id_lists), dtype=np.int64)
    for i, ids in enumerate(id_lists):
        ids = ids[:max_len] or [1]
        mat[i, : len(ids)] = ids
        lengths[i] = len(ids)
    return mat, lengths


def _word2vec_matrices(texts, kv, max_len):
    word2idx = {w: i + 2 for i, w in enumerate(kv.index_to_key)}
    id_lists = [[word2idx.get(w, 1) for w in t.split()] for t in texts]
    return _pad_ids(id_lists, max_len, pad_id=0), word2idx


def _embedding_matrix_from_kv(kv, word2idx):
    dim = kv.vector_size
    weight = np.zeros((len(word2idx) + 2, dim), dtype=np.float32)
    for w, i in word2idx.items():
        weight[i] = kv[w]
    rng = np.random.default_rng(SEED)
    weight[1] = rng.normal(0, 0.01, dim).astype(np.float32)  # UNK
    return weight


def _build_bilstm(input_dim: int, n_classes: int, pack: bool, embed_weight=None):
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence

    class BiLSTMClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.pack = pack
            self.use_embed = embed_weight is not None
            if self.use_embed:
                self.embedding = nn.Embedding.from_pretrained(
                    torch.tensor(embed_weight), freeze=False, padding_idx=0
                )
                in_dim = embed_weight.shape[1]
            else:
                self.embedding = None
                in_dim = input_dim
            self.lstm = nn.LSTM(
                in_dim, BILSTM["hidden_dim"], num_layers=1, batch_first=True, bidirectional=True
            )
            self.dropout = nn.Dropout(BILSTM["dropout"])
            self.fc = nn.Linear(2 * BILSTM["hidden_dim"], n_classes)

        def forward(self, x, lengths, mask=None):
            emb = self.dropout(self.embedding(x)) if self.use_embed else self.dropout(x)
            if self.pack:
                emb = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (hidden, _) = self.lstm(emb)
            joined = torch.cat([hidden[-2], hidden[-1]], dim=1)
            return self.fc(self.dropout(joined))

    return BiLSTMClassifier()


def _train_bilstm(model, train_batches, val_batches, test_batches, y_fit, y_val, y_test,
                  n_classes, epochs, lr, patience, batch_size, seed, device, method_label):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    def make_loader(batches, labels, shuffle):
        ids, lengths = batches
        ds = TensorDataset(
            torch.from_numpy(ids),
            torch.from_numpy(lengths),
            torch.from_numpy(np.asarray(labels, dtype=np.int64)),
        )
        gen = torch.Generator().manual_seed(seed)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen)

    fit_loader = make_loader(train_batches, y_fit, True)
    val_loader = make_loader(val_batches, y_val, False)
    test_loader = make_loader(test_batches, y_test, False)

    def run_epoch(dl, train):
        model.train(train)
        total, correct, count = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for ids, lengths, yb in dl:
                ids, yb = ids.to(device), yb.to(device)
                logits = model(ids, lengths)
                loss = loss_fn(logits, yb)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                total += loss.item() * len(yb)
                correct += (logits.argmax(1) == yb).sum().item()
                count += len(yb)
        return total / max(count, 1), correct / max(count, 1)

    history, best_state, best_val, wait = [], None, float("inf"), 0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(fit_loader, True)
        va_loss, va_acc = run_epoch(val_loader, False)
        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc,
        })
        print(
            f"  epoch {epoch}/{epochs} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
        )
        if va_loss < best_val:
            best_val, wait = va_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print("  early stopping")
                break
    train_time = time.perf_counter() - start
    model.load_state_dict(best_state)
    model.eval()
    start = time.perf_counter()
    probs = []
    with torch.no_grad():
        for ids, lengths, _ in test_loader:
            logits = model(ids.to(device), lengths)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    infer_time = time.perf_counter() - start
    metrics = compute_metrics(y_test, np.concatenate(probs), method_label, train_time, infer_time)
    metrics.update({
        "history": history,
        "epochs_run": len(history),
        "n_params": sum(p.numel() for p in model.parameters()),
    })
    return metrics


def _prepare_word2vec_sequences(dataset_dict, prep: PreprocessConfig, max_len_cap: int):
    from sklearn.model_selection import train_test_split

    y_train = dataset_dict["train_label"]
    fit_idx, val_idx = train_test_split(
        np.arange(len(y_train)), test_size=0.1, stratify=y_train, random_state=SEED
    )
    fit_text = [dataset_dict["train_text"][i] for i in fit_idx]
    val_text = [dataset_dict["train_text"][i] for i in val_idx]
    test_text = dataset_dict["test_text"]

    extract_and_save_features(
        dataset_dict, ExperimentConfig(dataset_dict["name"], prep, "word2vec")
    )
    folder = cache_dir_for(dataset_dict["name"], "word2vec", prep.tag(), FEATURE_DIR)
    kv = train_or_load_word2vec(dataset_dict["train_text"], folder, SEED)
    lengths = [len(t.split()) for t in fit_text]
    max_len = _resolve_max_len(lengths, prep, max_len_cap)
    (fit_ids, fit_len), word2idx = _word2vec_matrices(fit_text, kv, max_len)
    (val_ids, val_len), _ = _word2vec_matrices(val_text, kv, max_len)
    (test_ids, test_len), _ = _word2vec_matrices(test_text, kv, max_len)
    weight = _embedding_matrix_from_kv(kv, word2idx)
    return {
        "fit": (fit_ids, fit_len),
        "val": (val_ids, val_len),
        "test": (test_ids, test_len),
        "y_fit": y_train[fit_idx],
        "y_val": y_train[val_idx],
        "embed_weight": weight,
        "input_dim": weight.shape[1],
        "max_len": max_len,
        "device": get_device(),
    }


def run_bilstm_pipeline(
    dataset_dict: dict,
    prep_cfg: PreprocessConfig | None = None,
    dl_embedding: str = "word2vec",
    epochs: int | None = None,
    seed: int = SEED,
    loss_plot_path: Path | None = None,
    show: bool = False,
) -> dict:
    if dl_embedding != "word2vec":
        raise ValueError("This bonus BiLSTM path supports only Word2Vec embeddings")
    prep_cfg = prep_cfg or dataset_dict.get("preprocess") or PreprocessConfig()
    epochs = epochs or (2 if quick_mode() else BILSTM["epochs"])
    data = _prepare_word2vec_sequences(dataset_dict, prep_cfg, BILSTM["max_len_cap"])
    model = _build_bilstm(
        data["input_dim"],
        dataset_dict["n_classes"],
        pack=prep_cfg.padding,
        embed_weight=data["embed_weight"],
    )
    label = "BiLSTM (word2vec)"
    print(
        f"[bilstm] {label} {prep_cfg.tag()} on {data['device']}: "
        f"input_dim={data['input_dim']}, max_len={data['max_len']}"
    )
    metrics = _train_bilstm(
        model,
        data["fit"],
        data["val"],
        data["test"],
        data["y_fit"],
        data["y_val"],
        dataset_dict["test_label"],
        dataset_dict["n_classes"],
        epochs,
        BILSTM["lr"],
        BILSTM["patience"],
        BILSTM["batch_size"],
        seed,
        data["device"],
        label,
    )
    metrics.update({
        "classifier": "bilstm",
        "input": dl_embedding,
        "input_dim": data["max_len"],
        "max_len": data["max_len"],
        "vocab_size": int(data["embed_weight"].shape[0]),
        "unk_rate": 0.0,
        "dl_embedding": dl_embedding,
    })
    if loss_plot_path is not None and metrics.get("history"):
        history = metrics["history"]
        epochs_axis = [h["epoch"] for h in history]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        axes[0].plot(epochs_axis, [h["train_loss"] for h in history], "o-", label="train")
        axes[0].plot(epochs_axis, [h["val_loss"] for h in history], "o-", label="validation")
        axes[0].set_title("Cross-entropy loss")
        axes[1].plot(epochs_axis, [h["train_acc"] for h in history], "o-", label="train")
        axes[1].plot(epochs_axis, [h["val_acc"] for h in history], "o-", label="validation")
        axes[1].set_title("Accuracy")
        for ax in axes:
            ax.set_xlabel("epoch")
            ax.legend()
        fig.suptitle(f"{dataset_dict['title']}: {label} ({prep_cfg.tag()})")
        fig.tight_layout()
        Path(loss_plot_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(loss_plot_path, dpi=120)
        if show:
            plt.show()
        plt.close(fig)
    return metrics


def run_dl_pipeline(
    dataset_dict: dict,
    dl_model: str,
    dl_embedding: str,
    prep_cfg: PreprocessConfig | None = None,
    loss_plot_path: Path | None = None,
    show: bool = False,
) -> dict | None:
    """Dispatch the bonus deep-learning path (Word2Vec + BiLSTM)."""
    if dl_model not in DL_MODELS:
        raise ValueError(f"Unknown dl_model '{dl_model}'")
    if dl_embedding not in DL_EMBEDDINGS:
        raise ValueError(f"Unknown dl_embedding '{dl_embedding}'")
    prep_cfg = prep_cfg or dataset_dict.get("preprocess") or PreprocessConfig()
    return run_bilstm_pipeline(
        dataset_dict, prep_cfg, dl_embedding, loss_plot_path=loss_plot_path, show=show
    )
