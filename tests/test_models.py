import pandas as pd
import pytest
import torch

from phishlens.char_encoding import MAX_LEN, encode_batch
from phishlens.charcnn import CharCNN
from phishlens.data import DATASET, split_dataset
from phishlens.evaluate import compute_metrics
from phishlens.predict import LEVELS, MODELS_DIR, Predictor


def test_compute_metrics():
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.8, 0.9, 0.3])  # 1 false alarm, 1 missed
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (1, 1, 1, 1)
    assert m["accuracy"] == m["precision"] == m["recall"] == m["f1"] == 0.5
    assert m["fpr"] == 0.5


def test_split_keeps_each_website_in_one_part():
    df = pd.DataFrame({
        "url": [f"https://site{i % 50}.com/{i}" for i in range(500)],
        "label": [i % 2 for i in range(500)],
        "domain": [f"site{i % 50}.com" for i in range(500)],
    })  # fmt: skip
    train, val, test = split_dataset(df)
    assert len(train) + len(val) + len(test) == 500
    parts = [set(p.domain) for p in (train, val, test)]
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])


def test_charcnn_output_shape():
    model = CharCNN().eval()
    ids = torch.from_numpy(encode_batch(["a.com", "paypal.com/login", "b.org"]))
    assert ids.shape == (3, MAX_LEN)
    assert model(ids).shape == (3,)


@pytest.mark.skipif(not (MODELS_DIR / "charcnn.onnx").exists(), reason="models not trained")
def test_predictor_scores_a_url():
    result = Predictor().score("https://www.example.com/")
    assert 0 <= result["risk_score"] <= 100
    assert result["risk_level"] in LEVELS
    assert 0 <= result["lightgbm"] <= 1 and 0 <= result["cnn"] <= 1


@pytest.mark.skipif(not (MODELS_DIR / "charcnn.onnx").exists(), reason="models not trained")
def test_predictor_zones():
    p = Predictor()
    assert [p.risk_level(s) for s in (0, 39, 40, 69, 70, 100)] == [
        LEVELS[0], LEVELS[0], LEVELS[1], LEVELS[1], LEVELS[2], LEVELS[2]
    ]  # fmt: skip


@pytest.mark.skipif(not (MODELS_DIR / "charcnn.onnx").exists(), reason="models not trained")
def test_predictor_rejects_invalid_url():
    with pytest.raises(ValueError):
        Predictor().score("mailto:someone@example.com")


def test_dataset_path_points_into_project():
    assert DATASET.parts[-3:] == ("data", "processed", "dataset.csv")
