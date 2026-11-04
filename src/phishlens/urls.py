"""Shared URL helpers.

Used by BOTH the dataset builder and (later) the feature extractor / live app, so every part
of the project cleans and splits URLs in exactly the same way.
"""

from urllib.parse import quote, urlsplit, urlunsplit

import tldextract

# Offline extractor: uses the public-suffix list bundled with tldextract instead of downloading
# it, so results are reproducible and the app never needs network access for this.
# include_psl_private_domains=True treats hosting suffixes like "r2.dev" as public suffixes,
# so "pub-123.r2.dev" counts as its own site rather than as part of "r2.dev".
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)

ALLOWED_SCHEMES = {"http", "https"}
DEFAULT_PORTS = {"http": ":80", "https": ":443"}
# Schemes written without "//" that would otherwise be mistaken for a host ("mailto:x@y.com").
NON_WEB_SCHEMES = ("mailto:", "javascript:", "data:", "tel:", "file:", "about:")
MAX_URL_LENGTH = 2000
# Characters left as-is when %-encoding; everything else (e.g. "é") becomes %XX.
# "%" is included so already-encoded URLs are not double-encoded.
_SAFE_URL_CHARS = "/:@!$&'()*+,;=?%-._~"


def normalize_url(url: str) -> str | None:
    """Clean a raw URL. Returns None if it is not a usable http(s) URL.

    - strips whitespace
    - adds "http://" when the scheme is missing
    - keeps only http / https
    - lowercases scheme and host (paths are case-sensitive, so they are kept as-is)
    - removes default ports, trailing "/" and "#fragment"; encodes non-ASCII characters
      (each of these was a collection artefact found during EDA, see docs/REPORT.md)
    """
    if not isinstance(url, str):
        return None
    url = url.strip()
    if not url or len(url) > MAX_URL_LENGTH or any(c.isspace() for c in url):
        return None
    if url.lower().startswith(NON_WEB_SCHEMES):
        return None
    if "://" not in url:
        url = "http://" + url

    try:
        parts = urlsplit(url)
        hostname = parts.hostname  # lowercased, without user:password@ and :port
    except ValueError:
        return None

    if parts.scheme.lower() not in ALLOWED_SCHEMES or not hostname or "." not in hostname:
        return None

    scheme, netloc = parts.scheme.lower(), parts.netloc.lower()
    # ":443" on https and ":80" on http are the defaults, so "site.com:443" IS "site.com".
    # Some sites add them in redirects; left in, they would become a fake "legit" signal.
    default_port = DEFAULT_PORTS[scheme]
    if netloc.endswith(default_port):
        netloc = netloc[: -len(default_port)]

    # Non-ASCII host -> punycode ("bücher.de" -> "xn--bcher-kva.de"), as browsers send it.
    if not hostname.isascii():
        try:
            netloc = netloc.replace(hostname, hostname.encode("idna").decode("ascii"))
        except UnicodeError:
            return None

    # Non-ASCII path/query -> %-encoding ("/café" -> "/caf%C3%A9"), as browsers send it.
    # The phishing feeds store URLs this way; our crawler saw raw characters.
    path = quote(parts.path, safe=_SAFE_URL_CHARS)
    query = quote(parts.query, safe=_SAFE_URL_CHARS)

    # Trailing "/" removed ("site.com/about/" -> "site.com/about"): it does not change the page,
    # but the phishing feeds strip it and websites keep it -> a fake "legit" signal.
    path = path.rstrip("/")

    # The "#fragment" is never sent to the server, so it is not part of the page address.
    # Our crawler removed it, the feeds did not -> dropped everywhere for consistency.
    return urlunsplit((scheme, netloc, path, query, ""))


def model_input(url: str) -> str:
    """The part of the URL the models are allowed to see.

    Removes the scheme ("http://", "https://") and a leading "www.". We collect legit and
    phishing URLs in different ways, so these parts would teach the model about OUR collection
    process instead of about phishing.
    """
    for prefix in ("https://", "http://"):
        if url.startswith(prefix):
            url = url[len(prefix) :]
            break
    if url.startswith("www."):
        url = url[4:]
    return url


def hostname(url: str) -> str:
    """'https://Login.PayPal.com:8080/x' -> 'login.paypal.com'"""
    return urlsplit(url).hostname or ""


def registered_domain(url: str) -> str:
    """The site a URL belongs to: 'https://login.paypal.co.uk/x' -> 'paypal.co.uk'.

    Used to group URLs by website (so the same site never lands in both train and test).
    For IP-address hosts it falls back to the IP itself.
    """
    host = hostname(url)
    return _EXTRACT(host).top_domain_under_public_suffix or host
