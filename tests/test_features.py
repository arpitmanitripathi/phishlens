import numpy as np
import pytest

from phishlens.char_encoding import MAX_LEN, PAD_ID, UNKNOWN_ID, VOCAB_SIZE, encode, encode_batch
from phishlens.features import FEATURE_NAMES, extract_features, features_table


def f(url):
    return extract_features(url)


# ---------------------------------------------------------------- features: shape and order


def test_always_29_features_in_fixed_order():
    assert len(FEATURE_NAMES) == 29 == len(set(FEATURE_NAMES))
    assert list(f("https://example.com/a?b=1")) == FEATURE_NAMES


def test_all_values_are_floats():
    assert all(isinstance(v, float) for v in f("https://example.com").values())


def test_scheme_and_www_do_not_change_features():
    # Phase 1 rule: http / https / www are collection bias -> invisible to the model.
    a = f("https://www.example.com/login")
    assert a == f("http://example.com/login") == f("example.com/login")


def test_raw_and_clean_url_give_same_features():
    assert f("  HTTPS://Example.com:443/about/#top ") == f("https://example.com/about")


def test_invalid_url_raises():
    with pytest.raises(ValueError):
        f("mailto:someone@example.com")


def test_features_table():
    table = features_table(["https://a.com", "http://b.org/x"])
    assert table.shape == (2, 29)
    assert list(table.columns) == FEATURE_NAMES


# ---------------------------------------------------------------- features: single values


def test_counts():
    x = f("https://shop.my-site.co.uk/a/b/c.php?x=1&y=2")
    assert x["domain_length"] == len("shop.my-site.co.uk")
    assert x["path_length"] == len("/a/b/c.php")
    assert x["num_path_segments"] == 3
    assert x["num_query_params"] == 2
    assert x["num_hyphens_in_domain"] == 1
    assert x["num_subdomains"] == 1  # "shop"
    assert x["path_has_file_extension"] == 1
    assert x["is_homepage"] == 0


def test_hyphens_counted_in_domain_only():
    # EDA: legit sites have many hyphens in article paths -> only the domain counts.
    assert f("https://bbc.co.uk/food/pasta-recipes-easy")["num_hyphens_in_domain"] == 0


def test_ip_host():
    x = f("http://192.168.1.1/login")
    assert x["has_ip"] == 1
    assert x["num_subdomains"] == 0
    assert x["suspicious_tld"] == 0


def test_tricks_in_the_address():
    assert f("http://paypal.com@evil.com/x")["has_at"] == 1
    assert f("https://https-secure-bank.com/")["https_in_domain"] == 1
    assert f("http://site.com:8080/x")["has_port"] == 1
    assert f("http://site.com/a//b")["has_double_slash_in_path"] == 1
    assert f("http://xn--pypal-4ve.com/")["has_punycode"] == 1


def test_lists():
    assert f("http://free-prize.tk/")["suspicious_tld"] == 1
    assert f("https://example.com/")["suspicious_tld"] == 0
    assert f("https://bit.ly/3abc")["is_shortener"] == 1
    assert f("https://example.com/login/verify-account")["num_suspicious_words"] == 3


def test_homepage():
    assert f("https://example.com/")["is_homepage"] == 1
    assert f("https://example.com/?q=1")["is_homepage"] == 0


# ---------------------------------------------------------------- features: brands


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://paypal.com/signin", 0),  # the real site
        ("https://paypal.co.uk/signin", 0),  # the real site, other country
        ("https://research.google/ai", 0),  # Google's own TLD
        ("http://secure-paypal.com.evil.xyz/", 1),  # brand on someone else's domain
        ("http://paypal-login.xyz/", 1),
        ("https://evil.com/paypal/login", 1),  # brand in the path
        ("https://purchase.com/", 0),  # "chase" inside another word does not count
        ("https://pineapple.com/", 0),  # neither does "apple"
    ],
)
def test_brand_in_url_not_domain(url, expected):
    assert f(url)["brand_in_url_not_domain"] == expected


@pytest.mark.parametrize(
    "url, expected",
    [
        ("http://paypa1.com/", 1),  # 1 -> l
        ("http://rnicrosoft-login.com/", 1),  # rn -> m
        ("http://faceb00k.com/", 1),  # 0 -> o
        ("http://netfllx.com/", 1),  # one typo
        ("https://paypal.com/", 0),  # the real word
        ("https://apply.com/", 0),  # "apple" is too short for the typo rule
        ("https://cloud.example.com/", 0),  # not "icloud": first letter differs
        ("https://finance.yahoo.com/", 0),  # not "binance"
        ("https://example.com/", 0),
    ],
)
def test_brand_lookalike(url, expected):
    assert f(url)["brand_lookalike"] == expected


# ---------------------------------------------------------------- features: randomness


def test_entropy_and_ratios():
    random_looking = f("http://x7kq9z2v.com/")
    normal = f("https://google.com/")
    assert random_looking["domain_entropy"] > normal["domain_entropy"]
    assert random_looking["digit_ratio_in_domain"] > 0 == normal["digit_ratio_in_domain"]
    assert normal["vowel_ratio_in_domain"] > random_looking["vowel_ratio_in_domain"]
    assert f("https://a.com/%41%42")["percent_encoded_ratio"] > 0


# ---------------------------------------------------------------- char encoding


def test_encode_shape_and_padding():
    ids = encode("https://www.example.com")
    assert ids.shape == (MAX_LEN,) and ids.dtype == np.int64
    assert (ids[len("example.com") :] == PAD_ID).all()
    assert ids.max() < VOCAB_SIZE


def test_encode_hides_scheme_www_and_case():
    assert (encode("https://www.PayPal.com/x") == encode("http://paypal.com/x")).all()


def test_encode_long_url_is_cut():
    ids = encode("https://example.com/" + "a" * 500)
    assert ids.shape == (MAX_LEN,) and (ids != PAD_ID).all()


def test_encode_unknown_character():
    # "%" is in the vocabulary, but a raw character the vocabulary lacks maps to UNKNOWN_ID.
    # After normalize_url every char is ASCII, so the vocabulary covers almost everything.
    assert UNKNOWN_ID not in encode("https://example.com/a%20b")


def test_encode_batch():
    assert encode_batch(["a.com", "b.org"]).shape == (2, MAX_LEN)
