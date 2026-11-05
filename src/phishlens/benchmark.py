"""Benchmark: why LightGBM? Compare 4 classic models on the same 29 features.

All models train on the TRAIN set and are compared on the VALIDATION set (the test set stays
untouched until the final evaluation in train.py). Also draws a learning curve: does LightGBM
still improve with more data, or would collecting more URLs not help?

Run:  uv run python -m phishlens.benchmark
"""

import time

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from phishlens.data import SEED, load_dataset, split_dataset
from phishlens.evaluate import (
    COLORS,
    MUTED,
    RESULTS,
    compute_metrics,
    markdown_table,
    plot_bars,
    plt,
    save_figure,
)
from phishlens.features import features_table

MODELS = {
    # Scaling matters for LR only (trees do not care about the size of numbers).
    "Logistic Regression": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000)
    ),
    "Decision Tree": lambda: DecisionTreeClassifier(max_depth=12, random_state=SEED),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
    "LightGBM": lambda: LGBMClassifier(n_estimators=300, random_state=SEED, verbose=-1),
}
FRACTIONS = (0.25, 0.50, 0.75, 1.00)


def main():
    train, val, _ = split_dataset(load_dataset())
    X_train, X_val = features_table(train.url), features_table(val.url)
    y_train, y_val = train.label, val.label
    print(f"train {len(train):,} | validation {len(val):,}")

    # --- 1. four models, same features ---
    results, timings = {}, {}
    for name, make in MODELS.items():
        start = time.perf_counter()
        model = make().fit(X_train, y_train)
        timings[name] = time.perf_counter() - start
        results[name] = compute_metrics(y_val, model.predict_proba(X_val)[:, 1])
        print(f"{name:20s} F1 {results[name]['f1']:.4f}  ({timings[name]:.1f}s)")

    table = markdown_table(results)
    plot_bars(
        {n: 100 * m["f1"] for n, m in results.items()},
        "Benchmark — F1 on the validation set (29 features)", "F1 (%)", "benchmark_f1.png",
    )  # fmt: skip

    # --- 2. learning curve (LightGBM) ---
    curve = {}
    for frac in FRACTIONS:
        part = train.sample(frac=frac, random_state=SEED) if frac < 1 else train
        model = MODELS["LightGBM"]().fit(X_train.loc[part.index], y_train.loc[part.index])
        curve[frac] = compute_metrics(y_val, model.predict_proba(X_val)[:, 1])["f1"]
        print(f"learning curve {frac:.0%} of train ({len(part):,} URLs): F1 {curve[frac]:.4f}")

    fig, ax = plt.subplots(figsize=(6, 3.6))
    sizes = [int(f * len(train)) for f in FRACTIONS]
    ax.plot(sizes, [100 * v for v in curve.values()], color=COLORS["model"], lw=2, marker="o",
            markersize=8)  # fmt: skip
    for x, v in zip(sizes, curve.values(), strict=True):
        ax.annotate(f"{100 * v:.1f}", (x, 100 * v), textcoords="offset points", xytext=(0, 9),
                    ha="center", color=MUTED, fontsize=9)  # fmt: skip
    ax.set_ylim(80, 100)  # full, honest scale: a zoomed axis makes 0.8 points look huge
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:,.0f}")
    ax.set_xlabel("training URLs")
    ax.set_ylabel("validation F1 (%)")
    ax.set_title("Learning curve — LightGBM")
    save_figure(fig, "learning_curve.png")

    RESULTS.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Benchmark (validation set)",
        "",
        f"Train {len(train):,} URLs, validation {len(val):,} URLs, 29 features.",
        "",
        table,
        "",
        "Training time: " + ", ".join(f"{n} {t:.1f} s" for n, t in timings.items()),
        "",
        "## Learning curve (LightGBM, validation F1)",
        "",
        "| Training URLs | F1 |",
        "|---|---|",
        *[f"| {s:,} ({f:.0%}) | {100 * v:.1f} % |" for s, f, v in
          zip(sizes, FRACTIONS, curve.values(), strict=True)],
    ]  # fmt: skip
    (RESULTS / "benchmark.md").write_text("\n".join(lines) + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
