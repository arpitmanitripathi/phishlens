"""Turn one URL into the 29 numbers LightGBM learns from.

The SAME function is used for training (Phase 3) and for the live app (Phase 5), so the model
always sees features computed in exactly the same way.

Rule (see docs/REPORT.md, section 7.3): features only look at the part of the URL that means
the same thing in the real world as in our dataset -> model_input() hides "http(s)://" and "www.".
"""

import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlsplit

import pandas as pd

from phishlens.urls import _EXTRACT, hostname, model_input, normalize_url

# ---------------------------------------------------------------------------------------------
# Word lists. These come from general knowledge and public reports (e.g. Spamhaus / Interisle
# "most abused TLDs", brand-phishing reports), NOT from our dataset: a list built by looking at
# our phishing URLs would just memorise our data.
# ---------------------------------------------------------------------------------------------

SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq",  # formerly free TLDs, heavily abused
    "xyz", "top", "club", "online", "site", "icu", "buzz", "shop", "live", "rest",
    "cyou", "sbs", "cfd", "click", "link", "work", "fun", "monster", "quest", "bond",
}  # fmt: skip

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "cutt.ly",
    "rebrand.ly", "shorturl.at", "tiny.cc", "rb.gy", "bit.do", "s.id", "t.ly", "v.gd",
}  # fmt: skip

SUSPICIOUS_WORDS = (
    "login", "signin", "sign-in", "logon", "verify", "verification", "account", "update",
    "secure", "banking", "confirm", "password", "webscr", "suspend", "unlock", "wallet",
    "billing", "invoice", "recover", "authenticate", "validate",
)  # fmt: skip

# Brands that are often impersonated. Matched as WHOLE tokens, so "chase" in "purchase" and
# "apple" in "pineapple" do not count.
BRANDS = (
    "paypal", "apple", "icloud", "microsoft", "office365", "outlook", "google", "gmail",
    "facebook", "instagram", "whatsapp", "netflix", "amazon", "ebay", "chase", "wellsfargo",
    "bankofamerica", "citibank", "usps", "dhl", "fedex", "linkedin", "dropbox", "adobe",
    "docusign", "coinbase", "binance", "metamask", "steam", "roblox", "telegram",
)  # fmt: skip

# Look-alike characters attackers swap in: "paypa1" -> "paypal", "rnicrosoft" -> "microsoft".
HOMOGLYPHS = (("rn", "m"), ("vv", "w"), ("0", "o"), ("1", "l"), ("3", "e"), ("5", "s"), ("@", "a"))

FILE_EXTENSION = re.compile(r"\.(php|html?|aspx?|jsp|cgi|exe|zip|apk|scr|js)$", re.IGNORECASE)
PERCENT_ENCODED = re.compile(r"%[0-9a-fA-F]{2}")
TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")
VOWELS = set("aeiou")

FEATURE_NAMES = [
    "url_length",
    "domain_length",
    "path_length",
    "num_dots",
    "num_hyphens_in_domain",
    "num_digits",
    "num_path_segments",
    "num_query_params",
    "num_special_chars",
    "has_at",
    "has_ip",
    "https_in_domain",
    "has_port",
    "has_double_slash_in_path",
    "path_has_file_extension",
    "num_subdomains",
    "suspicious_tld",
    "is_shortener",
    "has_punycode",
    "num_suspicious_words",
    "brand_in_url_not_domain",
    "brand_lookalike",
    "domain_entropy",
    "digit_ratio_in_domain",
    "longest_token_length",
    "path_entropy",
    "percent_encoded_ratio",
    "vowel_ratio_in_domain",
    "is_homepage",
]


# ---------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------


def _entropy(text: str) -> float:
    """Shannon entropy: how 'random' a string looks. 'aaaa' -> 0, 'x7kq9z' -> high."""
    if not text:
        return 0.0
    n = len(text)
    return -sum(c / n * math.log2(c / n) for c in Counter(text).values())


def _tokens(text: str) -> list[str]:
    """'secure-PayPal.com/log_in' -> ['secure', 'paypal', 'com', 'log', 'in']"""
    return [t for t in TOKEN_SPLIT.split(text.lower()) if t]


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _edit_distance_at_most_1(a: str, b: str) -> bool:
    """True if a can become b with at most one insert, delete or substitute."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = j = edits = 0
    while i < len(a) and j < len(b):
        if a[i] != b[j]:
            edits += 1
            if edits > 1:
                return False
            if len(a) == len(b):
                i += 1  # substitute
            j += 1  # (insert into a) or substitute
        else:
            i += 1
            j += 1
    return edits + (len(b) - j) <= 1


def _looks_like_brand(token: str) -> bool:
    """'paypa1' / 'rnicrosoft' / 'facebok' -> True; 'paypal' (the real word) -> False."""
    if token in BRANDS:
        return False
    swapped = token
    for fake, real in HOMOGLYPHS:
        swapped = swapped.replace(fake, real)
    if swapped in BRANDS:
        return True
    # One typo away. Only for long brands (short ones: "apple" -> "apply" is a real word) and
    # only with the same first letter ("cloud" is not "icloud", "finance" is not "binance").
    return any(
        len(b) >= 6 and token[0] == b[0] and _edit_distance_at_most_1(token, b) for b in BRANDS
    )


# ---------------------------------------------------------------------------------------------
# The feature extractor
# ---------------------------------------------------------------------------------------------


def extract_features(url: str) -> dict[str, float]:
    """URL -> {feature name: value}, always the 29 features of FEATURE_NAMES in that order.

    Raises ValueError if the URL is not a usable http(s) URL.
    """
    clean = normalize_url(url)  # no-op for dataset URLs (already clean), needed for live input
    if clean is None:
        raise ValueError(f"not a valid http(s) URL: {url!r}")

    visible = model_input(clean)  # "paypal.com/login?x=1" (no scheme, no "www.")
    parts = urlsplit(clean)
    host = hostname(clean).removeprefix("www.")
    path, query = parts.path, parts.query
    is_ip = _is_ip(host)
    ext = _EXTRACT(host)
    site_name = ext.domain  # "paypal" in "login.paypal.co.uk"

    try:
        has_port = parts.port is not None
    except ValueError:  # port like ":99999" or ":abc" -> still "has a port"
        has_port = True

    visible_tokens = _tokens(visible)
    host_tokens = _tokens(host)
    host_letters = [c for c in host if c.isalpha()]
    lower = visible.lower()

    brand_mentioned = any(b in visible_tokens for b in BRANDS)
    # The brand's own site: "paypal.co.uk", or a brand TLD like "research.google".
    owns_brand = site_name in BRANDS or ext.suffix in BRANDS

    features = {
        # --- size and counts ---
        "url_length": len(visible),
        "domain_length": len(host),
        "path_length": len(path),
        "num_dots": visible.count("."),
        "num_hyphens_in_domain": host.count("-"),
        "num_digits": sum(c.isdigit() for c in visible),
        "num_path_segments": len([s for s in path.split("/") if s]),
        "num_query_params": len([p for p in query.split("&") if p]),
        "num_special_chars": sum(not c.isalnum() and c not in "./-" for c in visible),
        # --- tricks in the address ---
        "has_at": int("@" in visible),
        "has_ip": int(is_ip),
        "https_in_domain": int("http" in host),
        "has_port": int(has_port),
        "has_double_slash_in_path": int("//" in path),
        "path_has_file_extension": int(bool(FILE_EXTENSION.search(path))),
        "num_subdomains": 0 if is_ip or not ext.subdomain else len(ext.subdomain.split(".")),
        # --- domain reputation (lists, not learned from our data) ---
        "suspicious_tld": int(not is_ip and ext.suffix.split(".")[-1] in SUSPICIOUS_TLDS),
        "is_shortener": int(host in SHORTENERS),
        "has_punycode": int("xn--" in host),
        # --- words and brands ---
        "num_suspicious_words": sum(w in lower for w in SUSPICIOUS_WORDS),
        "brand_in_url_not_domain": int(brand_mentioned and not owns_brand),
        "brand_lookalike": int(any(_looks_like_brand(t) for t in host_tokens)),
        # --- how random / unusual the text looks ---
        "domain_entropy": _entropy(host),
        "digit_ratio_in_domain": sum(c.isdigit() for c in host) / len(host),
        "longest_token_length": max((len(t) for t in visible_tokens), default=0),
        "path_entropy": _entropy(path),
        "percent_encoded_ratio": 3 * len(PERCENT_ENCODED.findall(visible)) / len(visible),
        "vowel_ratio_in_domain": (
            sum(c in VOWELS for c in host_letters) / len(host_letters) if host_letters else 0.0
        ),
        "is_homepage": int(not path and not query),
    }
    return {name: float(features[name]) for name in FEATURE_NAMES}


def features_table(urls) -> pd.DataFrame:
    """Many URLs -> DataFrame with one row per URL and the 29 feature columns (in order)."""
    return pd.DataFrame([extract_features(u) for u in urls], columns=FEATURE_NAMES)
