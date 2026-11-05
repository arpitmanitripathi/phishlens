"""The character-level CNN: reads a URL as a sequence of characters.

    ids (200)  ->  Embedding  ->  3 convolutions  ->  max-pool  ->  dropout  ->  1 output
                 (each char =     (look at windows     (keep the strongest      (phishing
                  32 numbers)      of 3, 4, 5 chars)    match per filter)        score)

A convolution filter is a small pattern detector sliding over the URL, e.g. one may fire on
"log" + "in", another on "-secure-", another on random digit runs. The model learns these
patterns itself; we never tell it what to look for.

Only used for TRAINING (needs PyTorch). The app runs the exported ONNX file instead.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from phishlens.char_encoding import MAX_LEN, PAD_ID, VOCAB_SIZE


class CharCNN(nn.Module):
    def __init__(self, embed_dim=32, num_filters=128, kernel_sizes=(3, 4, 5), dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=PAD_ID)
        self.convs = nn.ModuleList(nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, ids):  # ids: (batch, MAX_LEN) -> logits: (batch,)
        x = self.embedding(ids).transpose(1, 2)  # (batch, embed_dim, MAX_LEN)
        pooled = [torch.relu(conv(x)).amax(dim=2) for conv in self.convs]
        x = self.dropout(torch.cat(pooled, dim=1))
        return self.out(x).squeeze(1)


def _device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def predict_proba(model: CharCNN, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    """Encoded URLs -> phishing probability (0..1) for each."""
    model.eval()
    device = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i : i + batch_size]).to(device)
            out.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(out)


def train_cnn(X_train, y_train, X_val, y_val, epochs=40, batch_size=256, lr=1e-3, patience=3,
              seed=42, log=print):  # fmt: skip
    """Train with early stopping: stop when validation loss has not improved for `patience`
    epochs, and keep the best version (prevents over-fitting = memorising the training set).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = _device()
    model = CharCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    Xv = torch.from_numpy(X_val).to(device)
    yv = torch.from_numpy(np.asarray(y_val, dtype=np.float32)).to(device)
    y_train = np.asarray(y_train, dtype=np.float32)

    best_loss, best_state, bad_epochs, history = float("inf"), None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(X_train))  # new random order every epoch
        total = 0.0
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            xb = torch.from_numpy(X_train[idx]).to(device)
            yb = torch.from_numpy(y_train[idx]).to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            val_loss = sum(
                loss_fn(model(Xv[i : i + 2048]), yv[i : i + 2048]).item() * len(Xv[i : i + 2048])
                for i in range(0, len(Xv), 2048)
            ) / len(Xv)
        history.append({"epoch": epoch, "train_loss": total / len(order), "val_loss": val_loss})
        log(f"  epoch {epoch:2d}  train loss {total / len(order):.4f}  val loss {val_loss:.4f}")

        if val_loss < best_loss - 1e-4:
            best_loss, bad_epochs = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log(f"  early stop: no improvement for {patience} epochs")
                break

    model.load_state_dict(best_state)
    return model, history


class _WithSigmoid(nn.Module):
    """Exported model returns a probability directly, so the app needs no extra maths."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ids):
        return torch.sigmoid(self.model(ids))


def export_onnx(model: CharCNN, path: Path) -> None:
    # Export runs on the CPU. On macOS, LightGBM and PyTorch each load their own OpenMP
    # library, and after LightGBM has run, PyTorch's multi-threaded CPU code can freeze
    # (a known issue). The export is tiny, so one thread is enough and avoids the problem.
    torch.set_num_threads(1)
    model = _WithSigmoid(model.cpu().eval())
    example = torch.zeros(1, MAX_LEN, dtype=torch.int64)
    torch.onnx.export(
        model, (example,), str(path), input_names=["ids"], output_names=["prob"],
        dynamic_axes={"ids": {0: "batch"}, "prob": {0: "batch"}}, dynamo=False,
    )  # fmt: skip
