# PhishLens

**Real-time phishing URL detection with an explainable LightGBM + CNN ensemble.**

Real-time phishing URL detector. A soft-voting ensemble of **LightGBM** (29 engineered URL
features) and a **character-level CNN** (raw URL text) scores a URL, and the app returns a
verdict, a risk score and the top reasons behind it.

> 🚧 Work in progress. See [PLAN.md](PLAN.md) for the build plan.

## Quick start (development)

```bash
uv sync                 # create .venv with Python 3.12 and install everything
uv run pytest           # run tests
```

## Run the app locally
```bash
uv run uvicorn app.main:app --reload
# open http://127.0.0.1:8000  (API docs at /docs)
```

