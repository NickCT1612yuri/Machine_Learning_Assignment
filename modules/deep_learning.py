"""
Bonus deep-learning pipeline: BiLSTM, TextCNN, and SetFit over three embedding
backends (Word2Vec, DistilBERT, BGE-small). All nine combinations are runnable;
awkward SetFit×Word2Vec / SetFit×DistilBERT use SentenceTransformer adapters.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from modules.classifiers import align_proba, compute_metrics
from modules.config import (
    BERT_BATCH_SIZE,
    BERT_MODEL,
    BGE_MODEL,
    BILSTM,
    DL_EMBEDDINGS,
    DL_MODELS,
    FEATURE_DIR,
    SEED,
    SETFIT,
    TEXTCNN,
    WORD2VEC,
    PreprocessConfig,
)
from modules.data_loader import quick_mode
from modules.feature_extraction import cache_dir_for, extract_and_save_features, train_or_load_word2vec
from modules.config import ExperimentConfig
from modules.utils import get_device

UNPADDED_MAX_LEN = 10
TRANSFORMER_DIM = {"distilbert": 768, "bge_small": 384}


# ---------------------------------------------------------------------------
# Sequence batching helpers
# ---------------------------------------------------------------------------
def _resolve_max_len(lengths, prep: PreprocessConfig, cap: int) -> int:
    if not prep.padding:
        return UNPADDED_MAX_LEN
    p95 = int(np.ceil(np.percentile(lengths, 95))) if len(lengths) else 20
    return int(min(cap, max(20, p95)))


def _pad_ids(id_lists, max_len, pad_id=0):
    mat = np.full((len(id_lists), max_len), pad_id, dtype=np.int64)
    lengths = np.ones(len(id_lists), dtype=np.int64)
    for i, ids in enumerate(id_lists):
        ids = (ids[:max_len] or [1])
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


def _transformer_token_batches(texts, model_id, prep: PreprocessConfig, max_len, device):
    """Return list of (hidden [B,T,H], mask [B,T]) mini-batches under no_grad."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    tok = AutoTokenizer.from_pretrained(model_id)
    enc_model = AutoModel.from_pretrained(model_id).to(device)
    enc_model.eval()
    batches = []
    with torch.no_grad():
        for start in range(0, len(texts), BERT_BATCH_SIZE):
            batch = texts[start : start + BERT_BATCH_SIZE]
            if prep.tokenization:
                enc = tok(batch, padding="max_length", truncation=True, max_length=max_len,
                          return_tensors="pt")
            else:
                # Whole-word ids without subword split.
                vocab = tok.get_vocab()
                unk, cls, sep, pad = (tok.unk_token_id, tok.cls_token_id,
                                      tok.sep_token_id, tok.pad_token_id)
                rows = [[cls] + [vocab.get(w, unk) for w in t.split()][: max_len - 2] + [sep]
                        for t in batch]
                ids = torch.full((len(rows), max_len), pad, dtype=torch.long)
                mask = torch.zeros((len(rows), max_len), dtype=torch.long)
                for i, r in enumerate(rows):
                    ids[i, : len(r)] = torch.tensor(r[:max_len])
                    mask[i, : len(r)] = 1
                enc = {"input_ids": ids, "attention_mask": mask}
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            if prep.padding:
                hidden = enc_model(input_ids=ids, attention_mask=mask).last_hidden_state
            else:
                hidden = enc_model(input_ids=ids, attention_mask=torch.ones_like(mask)).last_hidden_state
            batches.append((hidden.detach(), mask.detach()))
    del enc_model
    return batches


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
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
            self.lstm = nn.LSTM(in_dim, BILSTM["hidden_dim"], num_layers=1,
                                batch_first=True, bidirectional=True)
            self.dropout = nn.Dropout(BILSTM["dropout"])
            self.fc = nn.Linear(2 * BILSTM["hidden_dim"], n_classes)

        def forward(self, x, lengths, mask=None):
            # x: token ids [B,T] if embedding, else dense [B,T,H]
            if self.use_embed:
                emb = self.dropout(self.embedding(x))
            else:
                emb = self.dropout(x)
            if self.pack:
                emb = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (hidden, _) = self.lstm(emb)
            joined = torch.cat([hidden[-2], hidden[-1]], dim=1)
            return self.fc(self.dropout(joined))

    return BiLSTMClassifier()


def _build_textcnn(input_dim: int, n_classes: int, embed_weight=None):
    import torch
    from torch import nn
    import torch.nn.functional as F

    class TextCNNClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.use_embed = embed_weight is not None
            if self.use_embed:
                self.embedding = nn.Embedding.from_pretrained(
                    torch.tensor(embed_weight), freeze=False, padding_idx=0
                )
                in_dim = embed_weight.shape[1]
            else:
                self.embedding = None
                in_dim = input_dim
            self.convs = nn.ModuleList([
                nn.Conv1d(in_dim, TEXTCNN["num_filters"], k) for k in TEXTCNN["filter_sizes"]
            ])
            self.dropout = nn.Dropout(TEXTCNN["dropout"])
            self.fc = nn.Linear(TEXTCNN["num_filters"] * len(TEXTCNN["filter_sizes"]), n_classes)

        def forward(self, x, lengths=None, mask=None):
            if self.use_embed:
                emb = self.embedding(x)  # [B,T,H]
            else:
                emb = x
            if mask is not None:
                emb = emb * mask.unsqueeze(-1).type_as(emb)
            # Conv1d expects [B,H,T]
            emb = emb.transpose(1, 2)
            pooled = []
            for conv in self.convs:
                h = F.relu(conv(emb))
                pooled.append(F.max_pool1d(h, kernel_size=h.shape[-1]).squeeze(-1))
            cat = torch.cat(pooled, dim=1)
            return self.fc(self.dropout(cat))

    return TextCNNClassifier()


# ---------------------------------------------------------------------------
# Training loop shared by BiLSTM / TextCNN
# ---------------------------------------------------------------------------
def _train_sequence_model(model, train_batches, val_batches, test_batches, y_fit, y_val, y_test,
                          n_classes, epochs, lr, patience, batch_size, seed, device, method_label):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    def make_loader(batches, labels, shuffle):
        # batches: list of tensors already on CPU for ids, or we stack
        if isinstance(batches, tuple) and len(batches) == 2 and isinstance(batches[0], np.ndarray):
            ids, lengths = batches
            ds = TensorDataset(
                torch.from_numpy(ids),
                torch.from_numpy(lengths),
                torch.from_numpy(np.asarray(labels, dtype=np.int64)),
            )
            kind = "ids"
        else:
            # list of (hidden, mask) covering the split in order — concatenate
            hidden = torch.cat([b[0].cpu() for b in batches], dim=0)
            mask = torch.cat([b[1].cpu() for b in batches], dim=0)
            lengths = mask.sum(1).long().clamp(min=1)
            ds = TensorDataset(hidden, lengths, mask, torch.from_numpy(np.asarray(labels, dtype=np.int64)))
            kind = "dense"
        gen = torch.Generator().manual_seed(seed)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen), kind

    fit_loader, kind = make_loader(train_batches, y_fit, True)
    val_loader, _ = make_loader(val_batches, y_val, False)
    test_loader, _ = make_loader(test_batches, y_test, False)

    def run_epoch(dl, train):
        model.train(train)
        total, correct, count = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for batch in dl:
                if kind == "ids":
                    ids, lengths, yb = batch
                    ids, yb = ids.to(device), yb.to(device)
                    logits = model(ids, lengths)
                else:
                    hidden, lengths, mask, yb = batch
                    hidden, mask, yb = hidden.to(device), mask.to(device), yb.to(device)
                    logits = model(hidden, lengths, mask)
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
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc})
        print(f"  epoch {epoch}/{epochs} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}")
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
        for batch in test_loader:
            if kind == "ids":
                ids, lengths, _ = batch
                logits = model(ids.to(device), lengths)
            else:
                hidden, lengths, mask, _ = batch
                logits = model(hidden.to(device), lengths, mask.to(device))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    infer_time = time.perf_counter() - start
    metrics = compute_metrics(y_test, np.concatenate(probs), method_label, train_time, infer_time)
    metrics.update({"history": history, "epochs_run": len(history),
                    "n_params": sum(p.numel() for p in model.parameters())})
    return metrics


def _prepare_sequence_data(dataset_dict, dl_embedding: str, prep: PreprocessConfig, max_len_cap: int):
    """Returns fit/val/test payloads + embed_weight (or None) + input_dim + max_len."""
    from sklearn.model_selection import train_test_split

    y_train = dataset_dict["train_label"]
    fit_idx, val_idx = train_test_split(
        np.arange(len(y_train)), test_size=0.1, stratify=y_train, random_state=SEED
    )
    fit_text = [dataset_dict["train_text"][i] for i in fit_idx]
    val_text = [dataset_dict["train_text"][i] for i in val_idx]
    test_text = dataset_dict["test_text"]
    device = get_device()

    if dl_embedding == "word2vec":
        # Ensure KV exists via feature cache.
        extract_and_save_features(
            dataset_dict,
            ExperimentConfig(dataset_dict["name"], prep, "word2vec"),
        )
        folder = cache_dir_for(dataset_dict["name"], "word2vec", prep.tag(), FEATURE_DIR)
        kv = train_or_load_word2vec(dataset_dict["train_text"], folder, SEED)
        lengths = [len(t.split()) for t in fit_text]
        max_len = max(_resolve_max_len(lengths, prep, max_len_cap), max(TEXTCNN["filter_sizes"]))
        (fit_ids, fit_len), word2idx = _word2vec_matrices(fit_text, kv, max_len)
        (val_ids, val_len), _ = _word2vec_matrices(val_text, kv, max_len)
        (test_ids, test_len), _ = _word2vec_matrices(test_text, kv, max_len)
        weight = _embedding_matrix_from_kv(kv, word2idx)
        return {
            "kind": "ids",
            "fit": (fit_ids, fit_len),
            "val": (val_ids, val_len),
            "test": (test_ids, test_len),
            "y_fit": y_train[fit_idx],
            "y_val": y_train[val_idx],
            "embed_weight": weight,
            "input_dim": weight.shape[1],
            "max_len": max_len,
            "device": device,
        }

    model_id = BERT_MODEL if dl_embedding == "distilbert" else BGE_MODEL
    # Approximate max_len from whitespace; transformers will truncate.
    lengths = [len(t.split()) + 2 for t in fit_text]
    max_len = max(
        _resolve_max_len(lengths, prep, min(max_len_cap, dataset_dict["max_len_cap"])),
        max(TEXTCNN["filter_sizes"]),
    )
    fit_b = _transformer_token_batches(fit_text, model_id, prep, max_len, device)
    val_b = _transformer_token_batches(val_text, model_id, prep, max_len, device)
    test_b = _transformer_token_batches(test_text, model_id, prep, max_len, device)
    return {
        "kind": "dense",
        "fit": fit_b,
        "val": val_b,
        "test": test_b,
        "y_fit": y_train[fit_idx],
        "y_val": y_train[val_idx],
        "embed_weight": None,
        "input_dim": TRANSFORMER_DIM[dl_embedding],
        "max_len": max_len,
        "device": device,
    }


def run_bilstm_pipeline(dataset_dict: dict, prep_cfg: PreprocessConfig | None = None,
                        dl_embedding: str = "word2vec", epochs: int | None = None,
                        seed: int = SEED, loss_plot_path: Path | None = None,
                        show: bool = False) -> dict:
    prep_cfg = prep_cfg or dataset_dict.get("preprocess") or PreprocessConfig()
    epochs = epochs or (2 if quick_mode() else BILSTM["epochs"])
    data = _prepare_sequence_data(dataset_dict, dl_embedding, prep_cfg, BILSTM["max_len_cap"])
    model = _build_bilstm(data["input_dim"], dataset_dict["n_classes"], pack=prep_cfg.padding,
                          embed_weight=data["embed_weight"])
    label = f"BiLSTM ({dl_embedding})"
    print(f"[bilstm] {label} {prep_cfg.tag()} on {data['device']}: "
          f"input_dim={data['input_dim']}, max_len={data['max_len']}")
    metrics = _train_sequence_model(
        model, data["fit"], data["val"], data["test"],
        data["y_fit"], data["y_val"], dataset_dict["test_label"],
        dataset_dict["n_classes"], epochs, BILSTM["lr"], BILSTM["patience"],
        BILSTM["batch_size"], seed, data["device"], label,
    )
    metrics.update({
        "classifier": "bilstm",
        "input": dl_embedding,
        "input_dim": data["max_len"],
        "max_len": data["max_len"],
        "vocab_size": 0 if data["embed_weight"] is None else int(data["embed_weight"].shape[0]),
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


def run_textcnn_pipeline(dataset_dict: dict, prep_cfg: PreprocessConfig | None = None,
                         dl_embedding: str = "word2vec", epochs: int | None = None,
                         seed: int = SEED) -> dict:
    prep_cfg = prep_cfg or dataset_dict.get("preprocess") or PreprocessConfig()
    epochs = epochs or (2 if quick_mode() else TEXTCNN["epochs"])
    data = _prepare_sequence_data(dataset_dict, dl_embedding, prep_cfg, TEXTCNN["max_len_cap"])
    model = _build_textcnn(data["input_dim"], dataset_dict["n_classes"],
                           embed_weight=data["embed_weight"])
    label = f"TextCNN ({dl_embedding})"
    print(f"[textcnn] {label} {prep_cfg.tag()} on {data['device']}: "
          f"input_dim={data['input_dim']}, max_len={data['max_len']}")
    metrics = _train_sequence_model(
        model, data["fit"], data["val"], data["test"],
        data["y_fit"], data["y_val"], dataset_dict["test_label"],
        dataset_dict["n_classes"], epochs, TEXTCNN["lr"], TEXTCNN["patience"],
        TEXTCNN["batch_size"], seed, data["device"], label,
    )
    metrics.update({
        "classifier": "textcnn",
        "input": dl_embedding,
        "input_dim": data["max_len"],
        "max_len": data["max_len"],
        "vocab_size": 0 if data["embed_weight"] is None else int(data["embed_weight"].shape[0]),
        "dl_embedding": dl_embedding,
    })
    return metrics


# ---------------------------------------------------------------------------
# SetFit (+ adapters)
# ---------------------------------------------------------------------------
def setfit_settings(n_classes: int) -> tuple[int, int]:
    few = n_classes <= 10
    return SETFIT["shots"][0 if few else 1], SETFIT["num_iterations"][0 if few else 1]


def _build_setfit_word2vec(dataset_dict, prep: PreprocessConfig):
    """SentenceTransformer adapter: mean-pool Word2Vec embeddings (custom module)."""
    import torch
    from torch import nn
    from sentence_transformers import SentenceTransformer

    extract_and_save_features(
        dataset_dict, ExperimentConfig(dataset_dict["name"], prep, "word2vec")
    )
    folder = cache_dir_for(dataset_dict["name"], "word2vec", prep.tag(), FEATURE_DIR)
    kv = train_or_load_word2vec(dataset_dict["train_text"], folder, SEED)
    word2idx = {w: i + 2 for i, w in enumerate(kv.index_to_key)}
    weight = _embedding_matrix_from_kv(kv, word2idx)

    class MeanWord2VecEncoder(nn.Module):
        def __init__(self, emb_weight, mapping):
            super().__init__()
            self.embedding = nn.Embedding.from_pretrained(
                torch.tensor(emb_weight), freeze=False, padding_idx=0
            )
            self.mapping = mapping
            self.word_embedding_dimension = int(emb_weight.shape[1])
            self.max_seq_length = 256

        def tokenize(self, texts, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            texts = ["" if t is None else str(t) for t in texts]
            return {"sentences": texts}

        def forward(self, features):
            device = self.embedding.weight.device
            sentences = features.get("sentences")
            if sentences is None:
                # Fallback if an upstream module already produced embeddings.
                return features
            batch_ids = []
            for s in sentences:
                ids = [self.mapping.get(w, 1) for w in str(s).split()] or [1]
                batch_ids.append(ids[: self.max_seq_length])
            max_len = max((len(x) for x in batch_ids), default=1)
            mat = torch.zeros(len(batch_ids), max_len, dtype=torch.long, device=device)
            mask = torch.zeros(len(batch_ids), max_len, dtype=torch.float, device=device)
            for i, ids in enumerate(batch_ids):
                if not ids:
                    ids = [1]
                mat[i, : len(ids)] = torch.tensor(ids, device=device)
                mask[i, : len(ids)] = 1.0
            emb = self.embedding(mat)
            pooled = (emb * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
            features["sentence_embedding"] = pooled
            return features

        def get_sentence_embedding_dimension(self):
            return self.word_embedding_dimension

        def get_embedding_dimension(self):
            return self.word_embedding_dimension

    return SentenceTransformer(modules=[MeanWord2VecEncoder(weight, word2idx)])


def _build_setfit_distilbert():
    from sentence_transformers import SentenceTransformer, models

    word = models.Transformer(BERT_MODEL, max_seq_length=128)
    dim = word.get_embedding_dimension() if hasattr(word, "get_embedding_dimension") else word.get_word_embedding_dimension()
    try:
        pool = models.Pooling(dim, pooling_mode="mean")
    except TypeError:
        pool = models.Pooling(dim, pooling_mode_mean_tokens=True)
    return SentenceTransformer(modules=[word, pool])


def build_setfit_body(backend: str, dataset_dict: dict, prep: PreprocessConfig):
    from setfit import SetFitModel
    from sklearn.linear_model import LogisticRegression

    head = LogisticRegression(max_iter=1000, random_state=SEED)
    if backend == "bge_small":
        return SetFitModel.from_pretrained(BGE_MODEL), BGE_MODEL, 384
    if backend == "distilbert":
        st = _build_setfit_distilbert()
        model = SetFitModel(model_body=st, model_head=head)
        return model, BERT_MODEL, 768
    raise ValueError(f"Unknown SetFit SentenceTransformer backend '{backend}'")


def _run_setfit_word2vec(dataset_dict: dict, shots: int, iterations: int, seed: int) -> dict:
    """
    SetFit-style few-shot adapter for Word2Vec: contrastive fine-tune of the
    embedding table on sentence pairs, then a logistic-regression head.
    (Avoids SentenceTransformer pair-collator incompatibilities with a custom module.)
    """
    import torch
    from torch import nn
    from sklearn.linear_model import LogisticRegression
    from datasets import Dataset
    from setfit import sample_dataset

    prep = dataset_dict.get("preprocess") or PreprocessConfig()
    extract_and_save_features(
        dataset_dict, ExperimentConfig(dataset_dict["name"], prep, "word2vec")
    )
    folder = cache_dir_for(dataset_dict["name"], "word2vec", prep.tag(), FEATURE_DIR)
    kv = train_or_load_word2vec(dataset_dict["train_text"], folder, seed)
    word2idx = {w: i + 2 for i, w in enumerate(kv.index_to_key)}
    weight = _embedding_matrix_from_kv(kv, word2idx)
    device = get_device()
    emb = nn.Embedding.from_pretrained(torch.tensor(weight), freeze=False, padding_idx=0).to(device)
    optimizer = torch.optim.Adam(emb.parameters(), lr=1e-3)

    train_ds = Dataset.from_dict({
        "text": list(dataset_dict["train_text"]),
        "label": [int(x) for x in dataset_dict["train_label"]],
    })
    few = sample_dataset(train_ds, label_column="label", num_samples=shots, seed=seed)
    texts = list(few["text"])
    labels = np.asarray(few["label"], dtype=np.int64)
    n_classes = dataset_dict["n_classes"]
    print(f"[setfit] word2vec-mean-adapter: {shots} shots x {n_classes} classes = {len(few)} labelled "
          f"examples, {iterations} pair iterations (custom contrastive loop)")

    def encode_batch(batch_texts):
        ids_list = [[word2idx.get(w, 1) for w in str(t).split()] or [1] for t in batch_texts]
        max_len = max(len(x) for x in ids_list)
        mat = torch.zeros(len(ids_list), max_len, dtype=torch.long, device=device)
        mask = torch.zeros(len(ids_list), max_len, dtype=torch.float, device=device)
        for i, ids in enumerate(ids_list):
            mat[i, : len(ids)] = torch.tensor(ids, device=device)
            mask[i, : len(ids)] = 1.0
        e = emb(mat)
        return (e * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)

    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    emb.train()
    for _ in range(max(1, iterations)):
        # Sample random pairs: same-label -> pull, different -> push.
        for _step in range(max(1, len(texts))):
            i, j = rng.integers(0, len(texts), size=2)
            va, vb = encode_batch([texts[i], texts[j]])
            target = torch.tensor([1.0 if labels[i] == labels[j] else -1.0], device=device)
            loss = nn.functional.cosine_embedding_loss(va.unsqueeze(0), vb.unsqueeze(0), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    emb.eval()
    with torch.no_grad():
        x_few = encode_batch(texts).cpu().numpy()
        x_test = []
        test_texts = list(dataset_dict["test_text"])
        for s in range(0, len(test_texts), 64):
            x_test.append(encode_batch(test_texts[s : s + 64]).cpu().numpy())
        x_test = np.concatenate(x_test, axis=0)
    head = LogisticRegression(max_iter=1000, random_state=seed)
    head.fit(x_few, labels)
    train_time = time.perf_counter() - start
    start = time.perf_counter()
    proba = align_proba(head.classes_, head.predict_proba(x_test), n_classes)
    infer_time = time.perf_counter() - start
    label = f"SetFit ({shots}-shot, word2vec)"
    metrics = compute_metrics(dataset_dict["test_label"], proba, label, train_time, infer_time)
    metrics.update({
        "classifier": "setfit",
        "input": f"{len(few)} labelled texts",
        "input_dim": int(weight.shape[1]),
        "shots_per_class": shots,
        "n_labelled": len(few),
        "num_iterations": iterations,
        "dl_embedding": "word2vec",
        "notes_extra": "custom Word2Vec contrastive adapter",
    })
    return metrics


def run_setfit_pipeline(dataset_dict: dict, shots_per_class: int | None = None,
                        dl_embedding: str = "bge_small", seed: int = SEED) -> dict | None:
    previous_tqdm = os.environ.get("TQDM_DISABLE")
    os.environ["TQDM_DISABLE"] = "1"
    prep = dataset_dict.get("preprocess") or PreprocessConfig()
    try:
        n_classes = dataset_dict["n_classes"]
        shots, iterations = setfit_settings(n_classes)
        shots = shots_per_class or shots
        if quick_mode():
            iterations = 2

        if dl_embedding == "word2vec":
            return _run_setfit_word2vec(dataset_dict, shots, iterations, seed)

        from datasets import Dataset, disable_progress_bars
        from setfit import Trainer, TrainingArguments, sample_dataset

        disable_progress_bars()
        train_ds = Dataset.from_dict({
            "text": list(dataset_dict["train_text"]),
            "label": [int(x) for x in dataset_dict["train_label"]],
        })
        few = sample_dataset(train_ds, label_column="label", num_samples=shots, seed=seed)
        model, backend_name, dim = build_setfit_body(dl_embedding, dataset_dict, prep)
        print(f"[setfit] {backend_name}: {shots} shots x {n_classes} classes = {len(few)} labelled "
              f"examples, {iterations} iterations")

        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(
                output_dir=tmp, batch_size=SETFIT["batch_size"],
                num_epochs=SETFIT["num_epochs"], num_iterations=iterations,
                seed=seed, report_to="none", save_strategy="no", show_progress_bar=False,
            )
            trainer = Trainer(model=model, args=args, train_dataset=few)
            start = time.perf_counter()
            trainer.train()
            train_time = time.perf_counter() - start

        start = time.perf_counter()
        proba = model.predict_proba(list(dataset_dict["test_text"]), batch_size=64, as_numpy=True)
        infer_time = time.perf_counter() - start
        classes = getattr(model.model_head, "classes_", np.arange(proba.shape[1]))
        proba = align_proba(classes, np.asarray(proba), n_classes)
        label = f"SetFit ({shots}-shot, {dl_embedding})"
        metrics = compute_metrics(dataset_dict["test_label"], proba, label, train_time, infer_time)
        metrics.update({
            "classifier": "setfit",
            "input": f"{len(few)} labelled texts",
            "input_dim": dim,
            "shots_per_class": shots,
            "n_labelled": len(few),
            "num_iterations": iterations,
            "dl_embedding": dl_embedding,
        })
        del model, trainer
        return metrics
    except Exception as exc:  # noqa: BLE001
        print(f"[setfit] skipped ({dl_embedding}): {type(exc).__name__}: {exc}")
        return None
    finally:
        if previous_tqdm is None:
            os.environ.pop("TQDM_DISABLE", None)
        else:
            os.environ["TQDM_DISABLE"] = previous_tqdm


def run_dl_pipeline(dataset_dict: dict, dl_model: str, dl_embedding: str,
                    prep_cfg: PreprocessConfig | None = None,
                    loss_plot_path: Path | None = None, show: bool = False) -> dict | None:
    """Dispatch one of the 9 (dl_model × dl_embedding) combinations."""
    if dl_model not in DL_MODELS:
        raise ValueError(f"Unknown dl_model '{dl_model}'")
    if dl_embedding not in DL_EMBEDDINGS:
        raise ValueError(f"Unknown dl_embedding '{dl_embedding}'")
    prep_cfg = prep_cfg or dataset_dict.get("preprocess") or PreprocessConfig()
    if dl_model == "bilstm":
        return run_bilstm_pipeline(dataset_dict, prep_cfg, dl_embedding,
                                   loss_plot_path=loss_plot_path, show=show)
    if dl_model == "textcnn":
        return run_textcnn_pipeline(dataset_dict, prep_cfg, dl_embedding)
    return run_setfit_pipeline(dataset_dict, dl_embedding=dl_embedding)
