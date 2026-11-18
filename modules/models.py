"""Classifiers: classical heads for stored features and the end-to-end TextCNN."""
import random
from collections import Counter

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch import nn

from .config import SEED, TEXTCNN
from .data import tokenize
from .evaluation import scores


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_classifier(name: str, param: float, scale: bool = False):
    """name in {'nb', 'logreg', 'svm'}; param is NB alpha or the C of LogReg/SVM."""
    if name == "nb":
        model = MultinomialNB(alpha=param)
    elif name == "logreg":
        model = LogisticRegression(C=param, max_iter=2000, random_state=SEED)
    else:
        model = LinearSVC(C=param, max_iter=5000, random_state=SEED)
    return make_pipeline(StandardScaler(), model) if scale else model


def predict_proba(model, X, n_classes):
    """Class probabilities aligned to all label ids, or None for LinearSVC (no probabilities)."""
    if not hasattr(model, "predict_proba"):
        return None
    proba = np.zeros((X.shape[0], n_classes))
    proba[:, model.classes_] = model.predict_proba(X)
    return proba


# ---------------------------------------------------------------- TextCNN
class TextCNN(nn.Module):
    """Kim (2014): embeddings -> parallel 1-D convolutions -> max-over-time pooling -> linear."""

    def __init__(self, vocab_size, n_classes, emb_dim, n_filters, widths, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(emb_dim, n_filters, w) for w in widths)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(n_filters * len(widths), n_classes)

    def forward(self, ids):
        x = self.embed(ids).transpose(1, 2)                      # (batch, emb_dim, seq_len)
        pooled = [torch.relu(conv(x)).amax(dim=2) for conv in self.convs]
        return self.out(self.dropout(torch.cat(pooled, dim=1)))


def build_vocab(texts):
    """Ids 0 = padding, 1 = unknown; keep frequent fit-split words only."""
    counts = Counter(w for t in texts for w in tokenize(t))
    words = [w for w, n in counts.most_common(TEXTCNN["vocab_size"] - 2) if n >= TEXTCNN["min_count"]]
    return {w: i + 2 for i, w in enumerate(words)}


def encode_ids(texts, vocab, max_len):
    """Truncate and right-pad every text to max_len token ids (fixed padding)."""
    ids = np.zeros((len(texts), max_len), dtype=np.int64)
    for row, text in enumerate(texts):
        seq = [vocab.get(w, 1) for w in tokenize(text)][:max_len] or [1]
        ids[row, :len(seq)] = seq
    return torch.from_numpy(ids)


def train_textcnn(corpus, batch_size=32, max_epochs=None, dev=None):
    """Adam + cross-entropy; keep the epoch with the best validation macro-F1 (early stopping)."""
    seed_everything()
    cfg = TEXTCNN
    max_epochs = max_epochs or cfg["max_epochs"]
    dev = dev or torch.device("cpu")
    vocab = build_vocab(corpus.text["fit"])
    X = {s: encode_ids(corpus.text[s], vocab, corpus.max_len) for s in corpus.text}
    y = {s: torch.from_numpy(corpus.y[s]) for s in corpus.y}
    model = TextCNN(len(vocab) + 2, len(corpus.labels), cfg["emb_dim"], cfg["n_filters"],
                    cfg["widths"], cfg["dropout"]).to(dev)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(SEED)

    def predict(split):
        model.eval()
        with torch.no_grad():
            logits = torch.cat([model(X[split][i:i + 256].to(dev)).cpu() for i in range(0, len(X[split]), 256)])
        proba = torch.softmax(logits, dim=1).numpy()
        return proba.argmax(1), proba

    history, best_f1, best_state, stale = [], -1.0, None, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        for idx in torch.randperm(len(y["fit"]), generator=generator).split(batch_size):
            xb, yb = X["fit"][idx].to(dev), y["fit"][idx].to(dev)
            optimiser.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimiser.step()
            total += loss.item() * len(idx)
        pred, proba = predict("val")
        history.append({"epoch": epoch, "train_loss": total / len(y["fit"]), **scores(corpus.y["val"], pred, proba)})
        if history[-1]["f1_macro"] > best_f1:
            best_f1, stale = history[-1]["f1_macro"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    pred, proba = predict("test")
    best_epoch = max(history, key=lambda h: h["f1_macro"])["epoch"]
    return pred, proba, history, {"best_epoch": best_epoch, "vocab_size": len(vocab)}
