"""Train the final models, choose the voting weight, evaluate ONCE on the test set, export.

    1. split by website (70 / 15 / 15)
    2. LightGBM on the 29 features      (early stopping on validation)
    3. Char-CNN on the raw characters   (early stopping on validation)
    4. voting weight w chosen on VALIDATION:  final = w * LightGBM + (1 - w) * CNN
    5. final scores on TEST (LightGBM alone, CNN alone, ensemble)
    6. export: models/lightgbm.joblib, models/charcnn.onnx, models/config.json
    7. reload the exported files and score hand-picked tricky URLs

Run:  uv run python -m phishlens.train
"""

import json
import time

import joblib
import numpy as np
import onnxruntime as ort
from lightgbm import LGBMClassifier, early_stopping

from phishlens.char_encoding import MAX_LEN, encode_batch
from phishlens.charcnn import export_onnx, predict_proba, train_cnn
from phishlens.data import SEED, load_dataset, split_dataset
from phishlens.evaluate import RESULTS, compute_metrics, markdown_table, plot_bars, plot_confusion
from phishlens.features import FEATURE_NAMES, features_table
from phishlens.predict import MODELS_DIR, Predictor

WEIGHTS = (0.3, 0.4, 0.5, 0.6, 0.7)
ZONES = {"suspicious": 40, "phishing": 70}  # risk score 0-39 safe, 40-69 suspicious, 70+ phishing

# Made-up URLs (not taken from the dataset) that test known weak spots.
# label 0 = legit, 1 = phishing-style. None of the phishing-style ones are real sites.
TRICKY_URLS = [
    ("https://accounts.google.com/signin", 0, "real login page"),
    ("https://www.paypal.com/signin", 0, "real login page"),
    ("https://login.microsoftonline.com", 0, "real login page, long domain"),
    ("https://github.com/login", 0, "real login page"),
    ("https://en.wikipedia.org/wiki/Phishing", 0, "the word phishing"),
    ("https://www.amazon.in/gp/your-account/order-history", 0, "account words, legit"),
    ("https://www.onlinesbi.sbi/personal/login.html", 0, "real bank, unusual TLD"),
    ("http://example.com", 0, "plain http"),
    ("https://someone.github.io/portfolio", 0, "free hosting (known weak spot)"),
    ("http://paypa1-secure-login.xyz/verify", 1, "look-alike + cheap TLD"),
    ("https://secure-paypal.com.account-update.top/signin", 1, "brand as subdomain"),
    ("http://192.168.10.5/wellsfargo/login.php", 1, "IP address host"),
    ("https://rnicrosoft-office365.com/auth", 1, "rn -> m look-alike"),
    ("https://amazon-order-cancel.online/refund?id=88231", 1, "brand + cheap TLD"),
    ("https://docs-google.com.verify-user.icu/login", 1, "brand hidden in subdomain"),
    ("https://login-microsoftonline-com.web.app", 1, "free hosting phishing"),
    ("https://my-account-help.com/billing/update", 1, "no brand, only words"),
]


def main():
    start = time.perf_counter()
    df = load_dataset()
    train, val, test = split_dataset(df)
    assert not set(train.domain) & set(test.domain), "a website is in both train and test"
    print(f"split: train {len(train):,} | validation {len(val):,} | test {len(test):,}")

    y_train, y_val, y_test = train.label.to_numpy(), val.label.to_numpy(), test.label.to_numpy()
    X_train, X_val, X_test = (features_table(d.url) for d in (train, val, test))
    C_train, C_val, C_test = (encode_batch(d.url) for d in (train, val, test))

    # --- LightGBM ---
    print("\nLightGBM")
    lgbm = LGBMClassifier(
        n_estimators=2000, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, random_state=SEED, verbose=-1,
    )  # fmt: skip
    lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[early_stopping(100, False)])
    print(f"  trees used: {lgbm.best_iteration_}")
    lgbm_val, lgbm_test = (lgbm.predict_proba(X)[:, 1] for X in (X_val, X_test))

    # --- Char-CNN ---
    print("\nChar-CNN")
    cnn, history = train_cnn(C_train, y_train, C_val, y_val, seed=SEED)
    cnn_val, cnn_test = predict_proba(cnn, C_val), predict_proba(cnn, C_test)

    # --- voting weight, chosen on VALIDATION only ---
    print("\nVoting weight (validation F1)")
    weight_f1 = {w: compute_metrics(y_val, w * lgbm_val + (1 - w) * cnn_val)["f1"] for w in WEIGHTS}
    for w, f1 in weight_f1.items():
        print(f"  w = {w:.1f}  F1 {f1:.4f}")
    best_w = max(WEIGHTS, key=lambda w: (round(weight_f1[w], 4), -abs(w - 0.5)))  # tie -> 0.5
    print(f"  chosen w = {best_w}")

    # --- final evaluation on TEST (first and only time the test set is used) ---
    ens_test = best_w * lgbm_test + (1 - best_w) * cnn_test
    test_metrics = {
        "LightGBM": compute_metrics(y_test, lgbm_test),
        "Char-CNN": compute_metrics(y_test, cnn_test),
        f"Ensemble (w = {best_w})": compute_metrics(y_test, ens_test),
    }
    table = markdown_table(test_metrics)
    print("\nTest set\n" + table)

    risk = np.round(100 * ens_test)
    zone = np.select([risk >= ZONES["phishing"], risk >= ZONES["suspicious"]], [2, 1], 0)
    zone_share = {
        cls: [float((zone[y_test == c] == z).mean()) for z in (0, 1, 2)]
        for c, cls in ((0, "legit"), (1, "phishing"))
    }

    # --- charts ---
    plot_confusion(test_metrics[f"Ensemble (w = {best_w})"], "Ensemble — test set",
                   "confusion_matrix.png")  # fmt: skip
    gain = lgbm.booster_.feature_importance(importance_type="gain")
    gain = dict(sorted(zip(FEATURE_NAMES, 100 * gain / gain.sum(), strict=True),
                       key=lambda kv: -kv[1]))  # fmt: skip
    plot_bars(dict(list(gain.items())[:15]), "LightGBM — top 15 features (share of total gain)",
              "% of gain", "feature_importance.png")  # fmt: skip

    # --- export ---
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(lgbm, MODELS_DIR / "lightgbm.joblib")
    export_onnx(cnn, MODELS_DIR / "charcnn.onnx")
    config = {
        "weight": best_w,
        "zones": ZONES,
        "feature_names": FEATURE_NAMES,
        "max_len": MAX_LEN,
        "feature_importance": gain,
    }
    (MODELS_DIR / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    # The exported ONNX model must give the same numbers as the PyTorch model.
    onnx_test = ort.InferenceSession(str(MODELS_DIR / "charcnn.onnx")).run(None, {"ids": C_test})[0]
    max_diff = float(np.abs(onnx_test - cnn_test).max())
    print(f"\nONNX vs PyTorch: max difference {max_diff:.2e}")
    assert max_diff < 1e-4

    # --- tricky URLs, scored through the exported files (exactly what the app will do) ---
    predictor = Predictor()
    tricky = [(url, label, note, predictor.score(url)) for url, label, note in TRICKY_URLS]
    print("\nTricky URLs")
    for url, label, _, r in tricky:
        print(
            f"  {'phish' if label else 'legit'}  {r['risk_score']:3d}  {r['risk_level']:10s} {url}"
        )

    # --- save results ---
    RESULTS.mkdir(parents=True, exist_ok=True)
    metrics_json = {
        "split": {"train": len(train), "validation": len(val), "test": len(test)},
        "lightgbm_trees": int(lgbm.best_iteration_),
        "cnn_history": history,
        "weight_validation_f1": weight_f1,
        "weight": best_w,
        "test": test_metrics,
        "zone_share_test": zone_share,
        "onnx_max_diff": max_diff,
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics_json, indent=2) + "\n")

    zone_rows = [
        f"| {cls} | " + " | ".join(f"{100 * s:.1f} %" for s in shares) + " |"
        for cls, shares in zone_share.items()
    ]
    tricky_rows = [
        f"| `{url}` | {'phishing' if label else 'legit'} | {r['lightgbm']:.2f} | {r['cnn']:.2f} "
        f"| {r['risk_score']} | {r['risk_level']} | {note} |"
        for url, label, note, r in tricky
    ]
    lines = [
        "# Final results (test set)",
        "",
        f"Split by website: train {len(train):,}, validation {len(val):,}, test {len(test):,}.",
        f"LightGBM trees: {lgbm.best_iteration_}. Char-CNN epochs: {len(history)}.",
        "",
        "## Voting weight (validation F1)",
        "",
        "| w (LightGBM) | " + " | ".join(f"{w:.1f}" for w in WEIGHTS) + " |",
        "|---" * (len(WEIGHTS) + 1) + "|",
        "| F1 | " + " | ".join(f"{100 * f:.2f} %" for f in weight_f1.values()) + " |",
        "",
        f"Chosen: **w = {best_w}**",
        "",
        "## Test set",
        "",
        table,
        "",
        "## Risk zones on the test set (ensemble)",
        "",
        "| Actual class | Looks safe (0-39) | Suspicious (40-69) | Phishing (70-100) |",
        "|---|---|---|---|",
        *zone_rows,
        "",
        "## Hand-picked tricky URLs (made up, not from the dataset)",
        "",
        "| URL | Expected | LightGBM | CNN | Risk | Level | Note |",
        "|---|---|---|---|---|---|---|",
        *tricky_rows,
    ]
    (RESULTS / "results.md").write_text("\n".join(lines) + "\n")
    print(f"\ndone in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
