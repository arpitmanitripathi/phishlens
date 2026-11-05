"""Load the trained models and score one URL. Used by the app (Phase 5) and by train.py.

Needs only LightGBM + ONNX Runtime (no PyTorch), so the deployed app stays small.
"""

import json
from pathlib import Path

import joblib
import onnxruntime as ort
import pandas as pd

from phishlens.char_encoding import encode
from phishlens.data import ROOT
from phishlens.features import FEATURE_NAMES, extract_features
from phishlens.urls import normalize_url

MODELS_DIR = ROOT / "models"
LEVELS = ("Looks safe", "Suspicious", "Phishing")


class Predictor:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.config = json.loads((models_dir / "config.json").read_text())
        self.lgbm = joblib.load(models_dir / "lightgbm.joblib")
        self.cnn = ort.InferenceSession(str(models_dir / "charcnn.onnx"))
        self.weight = self.config["weight"]
        self.zones = self.config["zones"]

    def risk_level(self, risk_score: int) -> str:
        if risk_score >= self.zones["phishing"]:
            return LEVELS[2]
        if risk_score >= self.zones["suspicious"]:
            return LEVELS[1]
        return LEVELS[0]

    def score(self, url: str) -> dict:
        """Raises ValueError for anything that is not a usable http(s) URL."""
        features = extract_features(url)
        X = pd.DataFrame([features], columns=FEATURE_NAMES)
        p_lgbm = float(self.lgbm.predict_proba(X)[0, 1])
        p_cnn = float(self.cnn.run(None, {"ids": encode(url)[None, :]})[0][0])

        # Soft voting: weighted average of the two probabilities.
        final = self.weight * p_lgbm + (1 - self.weight) * p_cnn
        risk_score = round(100 * final)
        return {
            "url": normalize_url(url),
            "risk_score": risk_score,
            "risk_level": self.risk_level(risk_score),
            "lightgbm": round(p_lgbm, 4),
            "cnn": round(p_cnn, 4),
            "features": features,
        }
