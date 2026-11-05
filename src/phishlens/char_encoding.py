"""Turn a URL into a fixed-length list of numbers for the Char-CNN.

The CNN does not get hand-made features: it reads the URL character by character and finds
its own patterns. Neural networks need numbers, so every character gets an id:

    "paypa1.com" -> [p, a, y, p, a, 1, ., c, o, m, 0, 0, 0, ...]   (ids, padded to 200)

Used by training (Phase 3) and by the live app (Phase 5): same code -> same numbers.
"""

import string

import numpy as np

from phishlens.urls import model_input, normalize_url

MAX_LEN = 200  # longer URLs are cut, shorter ones padded (most URLs are well under 200)
PAD_ID = 0  # "empty" position after the end of the URL
UNKNOWN_ID = 1  # any character not in the vocabulary

# 26 letters + 10 digits + 32 punctuation marks = 68 characters, ids 2..69.
VOCAB = string.ascii_lowercase + string.digits + string.punctuation
CHAR_TO_ID = {c: i + 2 for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB) + 2  # + PAD + UNKNOWN = 70


def encode(url: str) -> np.ndarray:
    """URL -> array of MAX_LEN ids. Lowercased: 'PayPal' and 'paypal' look the same to the CNN.

    Raises ValueError if the URL is not a usable http(s) URL.
    """
    clean = normalize_url(url)
    if clean is None:
        raise ValueError(f"not a valid http(s) URL: {url!r}")
    text = model_input(clean).lower()[:MAX_LEN]
    ids = np.full(MAX_LEN, PAD_ID, dtype=np.int64)
    ids[: len(text)] = [CHAR_TO_ID.get(c, UNKNOWN_ID) for c in text]
    return ids


def encode_batch(urls) -> np.ndarray:
    """Many URLs -> 2-D array of shape (number of URLs, MAX_LEN)."""
    return np.stack([encode(u) for u in urls])
