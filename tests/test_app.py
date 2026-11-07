import pytest
from fastapi.testclient import TestClient

from app.main import app
from phishlens.predict import MODELS_DIR

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "charcnn.onnx").exists(), reason="models not trained"
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # "with" runs startup -> models are loaded
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_head_requests_allowed(client):
    assert client.head("/").status_code == 200
    assert client.head("/health").status_code == 200


def test_home_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "PhishLens" in r.text


def test_check_phishing(client):
    r = client.post("/api/check", json={"url": "http://paypa1-secure-login.xyz/verify"})
    assert r.status_code == 200
    body = r.json()
    assert body["risk_level"] == "Phishing"
    assert 70 <= body["risk_score"] <= 100
    assert 1 <= len(body["reasons"]) <= 3
    assert set(body["model_scores"]) == {"lightgbm", "cnn"}


def test_check_returns_cleaned_url(client):
    r = client.post("/api/check", json={"url": "  HTTPS://Example.com/a/#top "})
    assert r.json()["url"] == "https://example.com/a"


@pytest.mark.parametrize("bad", ["mailto:a@b.com", "not a url", "ftp://x.com/file"])
def test_check_rejects_non_web_addresses(client, bad):
    r = client.post("/api/check", json={"url": bad})
    assert r.status_code == 422
    assert "website address" in r.json()["detail"]


def test_check_rejects_empty_and_too_long(client):
    assert client.post("/api/check", json={"url": ""}).status_code == 422
    assert client.post("/api/check", json={"url": "a" * 3000}).status_code == 422
    assert client.post("/api/check", json={}).status_code == 422
