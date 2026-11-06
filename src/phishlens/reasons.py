"""Turn a prediction into up to 3 plain-English reasons.

How it works:
1. RED-FLAG RULES: each rule looks at one feature and, if it is a red flag for this URL, writes
   a message ("Mentions 'paypal' but the website is not PayPal's").
2. ORDER BY WHAT MATTERED FOR THIS URL: LightGBM can tell, for one URL, how much each feature
   pushed its score up or down (SHAP contributions, `pred_contrib=True`). Red flags are sorted
   by that push, and flags that did NOT push this URL towards phishing are left out, so every
   reason shown really influenced the result.
   Exception, KNOWN TRICKS (IP address, '@', look-alike brand, shortener, punycode, 'https' in the
   name): they are rare in the training data, so LightGBM gives them little weight, but each one
   is a definite, well-known trick. When present they are always listed first.
3. The Char-CNN cannot point to single features. When it is the one raising the alarm, a
   general reason about the character pattern is added.

Thresholds for "unusually long / many" are set where fewer than ~5 % of legit training URLs are
(95th percentile of legit URLs, see docs/REPORT.md).
"""

from urllib.parse import urlsplit

from phishlens.features import BRANDS, FILE_EXTENSION, SUSPICIOUS_WORDS, looks_like_brand, tokens
from phishlens.urls import hostname, model_input, normalize_url, split_host

MAX_REASONS = 3
CNN_ALARM = 0.7  # CNN probability above which its own reason is added
SAFE_MESSAGE = "No strong phishing signs found in this address."
CNN_MESSAGE = (
    "The character pattern of this address closely resembles known phishing URLs "
    "(deep-learning model)."
)


def _details(url: str) -> dict:
    """Facts about the URL that make messages specific (which TLD, which brand, which words)."""
    host = hostname(url).removeprefix("www.")
    ext = split_host(host)
    visible = model_input(url).lower()
    url_tokens = tokens(visible)
    return {
        "host": host,
        "tld": ext.suffix.split(".")[-1] if ext.suffix else "",
        "brands": [b for b in BRANDS if b in url_tokens],
        "lookalikes": [t for t in tokens(host) if looks_like_brand(t)],
        "words": [w for w in SUSPICIOUS_WORDS if w in visible],
        "extension": (FILE_EXTENSION.search(urlsplit(url).path) or [""])[0],
    }


def _quote(items):
    return ", ".join(f"'{i}'" for i in items[:3])


KNOWN_TRICKS = (
    "has_ip", "has_at", "brand_lookalike", "is_shortener", "has_punycode", "https_in_domain",
)  # fmt: skip

# feature -> (is it a red flag?, message). `f` = feature values, `d` = details above.
RULES = {
    "has_ip": (
        lambda f, d: f["has_ip"] == 1,
        lambda f, d: "Uses a raw IP address instead of a website name.",
    ),
    "has_at": (
        lambda f, d: f["has_at"] == 1,
        lambda f, d: "Contains '@': browsers ignore everything before it, hiding the real site.",
    ),
    "brand_lookalike": (
        lambda f, d: f["brand_lookalike"] == 1,
        lambda f, d: (
            f"{_quote(d['lookalikes'])} looks like a well-known brand name "
            "with letters swapped or misspelled."
        ),
    ),
    "brand_in_url_not_domain": (
        lambda f, d: f["brand_in_url_not_domain"] == 1,
        lambda f, d: (
            f"Mentions {_quote(d['brands'])}, but the website ({d['host']}) "
            "does not belong to that brand."
        ),
    ),
    "https_in_domain": (
        lambda f, d: f["https_in_domain"] == 1,
        lambda f, d: "Puts 'http' or 'https' inside the website name to look secure.",
    ),
    "is_shortener": (
        lambda f, d: f["is_shortener"] == 1,
        lambda f, d: "Shortened link: the real destination is hidden.",
    ),
    "has_punycode": (
        lambda f, d: f["has_punycode"] == 1,
        lambda f, d: "Uses international characters that can imitate normal letters.",
    ),
    "suspicious_tld": (
        lambda f, d: f["suspicious_tld"] == 1,
        lambda f, d: f"Ends in '.{d['tld']}', a domain ending often abused for phishing.",
    ),
    "num_suspicious_words": (
        lambda f, d: f["num_suspicious_words"] >= 1,
        lambda f, d: f"Contains words often used in phishing: {_quote(d['words'])}.",
    ),
    "domain_length": (
        lambda f, d: f["domain_length"] > 21,
        lambda f, d: f"Unusually long website name ({int(f['domain_length'])} characters).",
    ),
    "longest_token_length": (
        lambda f, d: f["longest_token_length"] > 16,
        lambda f, d: (
            f"Contains a very long unbroken string "
            f"({int(f['longest_token_length'])} characters), typical of generated addresses."
        ),
    ),
    "domain_entropy": (
        lambda f, d: f["domain_entropy"] > 3.9,
        lambda f, d: "The website name looks random rather than like real words.",
    ),
    "digit_ratio_in_domain": (
        lambda f, d: f["digit_ratio_in_domain"] > 0.2 and f["has_ip"] == 0,
        lambda f, d: "The website name is full of digits.",
    ),
    "num_hyphens_in_domain": (
        lambda f, d: f["num_hyphens_in_domain"] >= 2,
        lambda f, d: (
            f"The website name has {int(f['num_hyphens_in_domain'])} hyphens "
            "(e.g. 'secure-login-bank'), a common phishing style."
        ),
    ),
    "num_subdomains": (
        lambda f, d: f["num_subdomains"] >= 2,
        lambda f, d: (
            f"Has {int(f['num_subdomains'])} subdomains, which can push the real "
            "website name out of view."
        ),
    ),
    "url_length": (
        lambda f, d: f["url_length"] > 100,
        lambda f, d: f"Very long address ({int(f['url_length'])} characters).",
    ),
    "has_port": (
        lambda f, d: f["has_port"] == 1,
        lambda f, d: "Uses an unusual port number.",
    ),
    "has_double_slash_in_path": (
        lambda f, d: f["has_double_slash_in_path"] == 1,
        lambda f, d: "Contains '//' in the path, often used to redirect to another site.",
    ),
    "path_has_file_extension": (
        lambda f, d: f["path_has_file_extension"] == 1,
        lambda f, d: f"Points directly to a '{d['extension']}' file, as many phishing kits do.",
    ),
}


def explain(url: str, features: dict, contributions: dict, risk_score: int, p_cnn: float):
    """Up to 3 reasons, most influential first.

    features      : the 29 feature values of this URL
    contributions : feature -> how much it pushed THIS URL's LightGBM score towards phishing
    """
    url = normalize_url(url) or url
    details = _details(url)
    flags = {
        name: message(features, details)
        for name, (is_flag, message) in RULES.items()
        if is_flag(features, details)
    }

    # 1. known tricks, always first
    reasons = [flags[name] for name in KNOWN_TRICKS if name in flags]
    if risk_score < 40:  # "Looks safe" zone: only mention definite tricks, if any
        return reasons[:MAX_REASONS] or [SAFE_MESSAGE]

    # 2. other red flags that pushed THIS URL towards phishing, strongest push first
    others = sorted(
        (name for name in flags if name not in KNOWN_TRICKS and contributions.get(name, 0) > 0),
        key=lambda name: -contributions[name],
    )
    reasons += [flags[name] for name in others]
    reasons = reasons[:MAX_REASONS]

    # 3. the CNN's alarm, if it is raising one and there is room
    if p_cnn >= CNN_ALARM and len(reasons) < MAX_REASONS:
        reasons.append(CNN_MESSAGE)
    if not reasons:  # suspicious, but no single clear sign
        reasons.append("Several small signs together make this address look unusual.")
    return reasons
