import pytest

from phishlens.urls import hostname, model_input, normalize_url, registered_domain


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("https://Example.COM/Path?A=1", "https://example.com/Path?A=1"),  # path case kept
        ("  example.com/login  ", "http://example.com/login"),  # scheme added, spaces stripped
        ("HTTP://Site.org", "http://site.org"),
    ],
)
def test_normalize_url_cleans(raw, expected):
    assert normalize_url(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "ftp://188.128.111.33/file.html",  # not http(s)
        "mailto:someone@example.com",
        "http://localhost/",  # no dot in host
        "http://exa mple.com/",  # whitespace inside
        "http://" + "a" * 2001 + ".com",  # too long
        None,
    ],
)
def test_normalize_url_rejects(raw):
    assert normalize_url(raw) is None


def test_model_input_hides_scheme_and_www():
    assert model_input("https://www.paypal.com/login") == "paypal.com/login"
    assert model_input("http://paypa1-login.xyz/") == "paypa1-login.xyz/"


def test_hostname_strips_credentials_and_port():
    assert hostname("http://me:pw@Evil.com:8080/x") == "evil.com"


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://login.paypal.co.uk/x", "paypal.co.uk"),
        ("https://www.google.com/", "google.com"),
        ("https://pub-8b4c.r2.dev/a.html", "pub-8b4c.r2.dev"),  # hosting suffix = own site
        ("http://192.168.1.1/login", "192.168.1.1"),  # IP falls back to itself
    ],
)
def test_registered_domain(url, expected):
    assert registered_domain(url) == expected
