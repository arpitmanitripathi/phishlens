"""Smoke test: every library the project depends on can be imported."""

import importlib

import pytest

LIBRARIES = [
    "numpy",
    "pandas",
    "tldextract",
    "sklearn",
    "lightgbm",
    "onnxruntime",
    "fastapi",
    "torch",
    "onnx",
    "requests",
    "bs4",
    "matplotlib",
    "antiphisher",  # our own package
]


@pytest.mark.parametrize("name", LIBRARIES)
def test_library_imports(name):
    importlib.import_module(name)
