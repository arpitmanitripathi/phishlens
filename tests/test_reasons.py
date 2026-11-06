import pytest

from phishlens.features import FEATURE_NAMES, extract_features
from phishlens.predict import MODELS_DIR, Predictor
from phishlens.reasons import CNN_MESSAGE, MAX_REASONS, SAFE_MESSAGE, explain


def reasons(url, risk=90, p_cnn=0.0, push=None):
    """Explain with a chosen per-feature push (default: every feature pushed a little)."""
    push = push or dict.fromkeys(FEATURE_NAMES, 0.1)
    return explain(url, extract_features(url), push, risk, p_cnn)


def test_safe_zone_gets_safe_message():
    assert reasons("https://example.com", risk=10) == [SAFE_MESSAGE]


def test_known_trick_shown_even_in_safe_zone():
    assert "IP address" in reasons("http://10.0.0.1/a", risk=10)[0]


def test_at_most_three_reasons():
    out = reasons("http://paypal.com@secure-login-verify-account.top:8080//x.php")
    assert len(out) == MAX_REASONS


def test_known_tricks_come_first():
    out = reasons("http://192.168.10.5/wellsfargo/login.php")
    assert "IP address" in out[0]


def test_messages_are_specific():
    out = " ".join(reasons("https://amazon-order-cancel.online/refund"))
    assert "'amazon'" in out and "'.online'" in out


def test_lookalike_names_the_fake_word():
    assert any("'paypa1'" in r for r in reasons("http://paypa1.com/"))


def test_flags_that_did_not_push_towards_phishing_are_left_out():
    push = dict.fromkeys(FEATURE_NAMES, -0.1)  # every feature pushed towards LEGIT
    out = reasons("https://my-account-help.com/billing/update", push=push)
    assert not any("words often used" in r for r in out)


def test_order_follows_the_push():
    url = "https://amazon-order-cancel.online/refund"
    push = dict.fromkeys(FEATURE_NAMES, 0.0)
    push.update(suspicious_tld=0.5, brand_in_url_not_domain=2.0, domain_length=0.1)
    out = reasons(url, push=push)
    assert "'amazon'" in out[0] and "'.online'" in out[1]


def test_cnn_reason_added_when_cnn_alarms():
    push = dict.fromkeys(FEATURE_NAMES, 0.0)
    assert reasons("https://plain-site.com/x", p_cnn=0.9, push=push) == [CNN_MESSAGE]


def test_suspicious_without_clear_sign_still_explained():
    push = dict.fromkeys(FEATURE_NAMES, 0.0)
    out = reasons("https://plain.com/x", risk=50, push=push)
    assert len(out) == 1 and out[0] != SAFE_MESSAGE


@pytest.mark.skipif(not (MODELS_DIR / "charcnn.onnx").exists(), reason="models not trained")
def test_predictor_returns_reasons():
    result = Predictor().score("http://paypa1-secure-login.xyz/verify")
    assert result["risk_level"] == "Phishing"
    assert 1 <= len(result["reasons"]) <= MAX_REASONS
