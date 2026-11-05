"""Metrics, tables and charts shared by benchmark.py and train.py."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)

from phishlens.data import ROOT  # noqa: E402

IMAGES = ROOT / "docs" / "images"
RESULTS = ROOT / "docs" / "results"

# Same look as the EDA charts.
COLORS = {"legit": "#2a78d6", "phishing": "#eb6834", "model": "#1baf7a"}
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": TEXT,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})  # fmt: skip


def compute_metrics(y_true, prob, threshold: float = 0.5) -> dict:
    """Probabilities -> the metrics we report. Phishing (1) is the 'positive' class.

    precision : of the URLs we flagged, how many were really phishing
    recall    : of the real phishing URLs, how many we caught
    f1        : balance of the two (main metric)
    f2        : like f1 but counts recall double (missing phishing is worse than a false alarm)
    fpr       : false-positive rate, share of legit URLs wrongly flagged
    """
    y_true = np.asarray(y_true)
    pred = (np.asarray(prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "f2": fbeta_score(y_true, pred, beta=2, zero_division=0),
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }  # fmt: skip


def markdown_table(rows: dict[str, dict], columns=("accuracy", "precision", "recall", "f1", "fpr")):
    """{'LightGBM': metrics, ...} -> markdown table with values in %."""
    head = "| Model | " + " | ".join(c.capitalize() if c != "fpr" else "FPR" for c in columns)
    lines = [head + " |", "|---" * (len(columns) + 1) + "|"]
    for name, m in rows.items():
        lines.append(f"| {name} | " + " | ".join(f"{100 * m[c]:.1f} %" for c in columns) + " |")
    return "\n".join(lines)


def save_figure(fig, name: str) -> Path:
    IMAGES.mkdir(parents=True, exist_ok=True)
    path = IMAGES / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion(m: dict, title: str, name: str) -> Path:
    cells = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cells, cmap="Blues", vmin=0, vmax=cells.max() * 1.3)
    ax.grid(False)
    labels = [["correct: legit", "false alarm"], ["missed phishing", "caught phishing"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cells[i, j]:,}\n{labels[i][j]}", ha="center", va="center",
                    color=TEXT, fontsize=10)  # fmt: skip
    ax.set_xticks([0, 1], ["legit", "phishing"])
    ax.set_yticks([0, 1], ["legit", "phishing"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title)
    return save_figure(fig, name)


def plot_bars(values: dict[str, float], title: str, xlabel: str, name: str, fmt="{:.1f}"):
    """Horizontal bars, one colour (a single series), value written at the end of each bar."""
    labels, vals = list(values)[::-1], list(values.values())[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(vals) + 1))
    ax.barh(labels, vals, color=COLORS["model"], height=0.6)
    for y, v in enumerate(vals):
        ax.text(v, y, " " + fmt.format(v), va="center", color=MUTED, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    return save_figure(fig, name)
