# PhishLens — Final Build Plan

## Goal
A real-time phishing URL detector built as a complete ML project:
own dataset → 29 engineered features + raw URL text → LightGBM + Char-CNN ensemble →
explained verdicts → FastAPI app → live deployment on Render.
Simple enough to explain every part in an interview.

## Final decisions

| Topic | Decision |
|---|---|
| ML type | Supervised learning, binary classification (phishing = 1, legit = 0) |
| Data sources | Phishing: OpenPhish + Phishing.Database. Legit: Tranco top sites + real links (with paths) crawled from top sites |
| Data size | 20,000 phishing + 20,000 legit = 40,000 URLs (minimum 10k each), balanced |
| Split | 70 / 15 / 15 → 28k train / 6k validation / 6k test, grouped by domain |
| Features | 29, from the URL text only (no WHOIS/DNS, server never visits the URL) |
| Benchmark | Logistic Regression, Decision Tree, Random Forest, LightGBM (comparison only) |
| Final models | **LightGBM** (29 features) + **Char-CNN** (raw URL, PyTorch → ONNX) |
| Ensemble | Soft weighted voting: `final = w × LightGBM + (1 − w) × CNN` |
| Weight | w ∈ {0.3, 0.4, 0.5, 0.6, 0.7}, best F1 on validation set |
| Metrics | Accuracy, precision, recall, F1, confusion matrix |
| Output | Verdict + risk level + risk score (0–100) + top 3 reasons |
| Risk zones | 0–39 Looks safe ✅ · 40–69 Suspicious ⚠️ · 70–100 Phishing 🚫 (tuned on validation) |
| Reasons | Red-flag features → human-readable messages, sorted by LightGBM feature importance |
| Backend | FastAPI + Uvicorn, ONNX Runtime for the CNN (no PyTorch in production) |
| Frontend | Plain HTML + JavaScript |
| Testing | pytest |
| Deployment | Docker → Render (free tier) |

## Features (29, URL text only; http/https scheme ignored to avoid a fake signal)
| # | Feature | # | Feature |
|---|---|---|---|
| 1 | url_length | 16 | num_subdomains |
| 2 | domain_length | 17 | suspicious_tld |
| 3 | path_length | 18 | is_shortener |
| 4 | num_dots | 19 | has_punycode (xn--) |
| 5 | num_hyphens_in_domain | 20 | num_suspicious_words |
| 6 | num_digits | 21 | brand_in_url_not_domain |
| 7 | num_path_segments | 22 | brand_lookalike |
| 8 | num_query_params | 23 | domain_entropy |
| 9 | num_special_chars | 24 | digit_ratio_in_domain |
| 10 | has_at | 25 | longest_token_length |
| 11 | has_ip | 26 | path_entropy |
| 12 | https_in_domain | 27 | percent_encoded_ratio |
| 13 | has_port | 28 | vowel_ratio_in_domain |
| 14 | has_double_slash_in_path | 29 | is_homepage (no path, no query) |
| 15 | path_has_file_extension (.php .html .exe) | | |

Features 7 and 26–29 are ideas taken from studying PhishX (own implementation).

EDA notes for Phase 2 (notebooks/eda.ipynb):
- Hyphens: legit URLs have MORE hyphens overall (path slugs) → count hyphens in the domain only.
- Punycode and %-encoding are equally common in both classes → expect little value; keep, let the model decide.
- Brand matching must avoid substrings inside other words (e.g. "chase" in "purchase").
- Default ports (:443/:80) were a collection artefact → removed in normalize_url().

## Architecture
```
PART A — TRAINING (offline, laptop)
  build_dataset.py → dataset.csv (url, label)
     ├─► features.py → 29 numbers → LightGBM ──┐
     └─► char encoding → Char-CNN ─────────────┤
                                               ▼
                  soft voting (w tuned on validation) → evaluate on test
                  → models/lightgbm.joblib, models/charcnn.onnx, models/config.json

PART B — SERVING (live, Render)
  Web page → POST /api/check → validate URL
     ├─► features.py (SAME code) → LightGBM → p1
     └─► char encoding (SAME code) → CNN (ONNX) → p2
     → final = w·p1 + (1−w)·p2 → risk score → zone + reasons
     → { verdict, risk_score, risk_level, reasons }
```
Key rule: feature and char-encoding code is shared by training and serving.

## Folder structure
```
phishlens/
├── data/raw/                  # downloaded lists (not in git)
├── data/processed/            # dataset.csv
├── src/phishlens/
│   ├── features.py            # URL → 29 numbers
│   ├── char_encoding.py       # URL → sequence of character ids
│   ├── charcnn.py             # CNN model definition
│   ├── benchmark.py           # 4-model comparison
│   ├── train.py               # LightGBM + CNN + voting weight + export
│   ├── evaluate.py            # metrics, confusion matrix, comparison table
│   └── predict.py             # load models, score, zone, reasons
├── scripts/build_dataset.py   # download + crawl + clean + merge
├── app/main.py                # FastAPI
├── app/static/index.html      # web page
├── models/                    # lightgbm.joblib, charcnn.onnx, config.json
├── notebooks/eda.ipynb        # data exploration
├── tests/
├── Dockerfile
├── requirements.txt
├── README.md
└── PLAN.md
```

## Project report
`docs/REPORT.md` — living report, updated at the end of every phase.
Scheme (http/https) may be shown to users as a separate rule-based note, outside the model score.

## Phases (~11 days at 3–4 h/day)
- [x] **0. Setup** (0.5d) — folders, virtualenv, requirements, `.gitignore`, git init
- [x] **1. Data** (2d) — download feeds, crawl legit links, clean, dedupe, balance, `dataset.csv`, EDA
- [ ] **2. Features** (1.5d) — `features.py` (29) + `char_encoding.py` + tests
- [ ] **3. Models** (3d) — benchmark 4 models, train LightGBM + Char-CNN, tune voting weight, test-set evaluation, ONNX export
- [ ] **4. Reasons** (1d) — top 3 human-readable reasons per verdict
- [ ] **5. App** (1.5d) — FastAPI `/api/check`, `/health`, web page
- [ ] **6. Ship** (1.5d) — tests, Dockerfile, README with results, new GitHub repo, deploy on Render
- [ ] **7. Cleanup** — delete old GitHub repo (only after new one is live)
- [ ] **8. Optional** (+1.5d) — Chrome extension

## Known limitations (to state in README)
- URL-only: phishing hosted on hacked real sites or free hosts (sites.google.com, *.github.io) can look clean.
- Feeds contain some noise (dead or mislabelled URLs).
- High test scores on URL datasets overstate real-world performance; we also test tricky hand-picked URLs.

## Later upgrades (not now)
Domain age (WHOIS), stacking, LLM second check for borderline URLs.

## Resume line (fill in real numbers at the end)
> Built a real-time phishing URL detector using a soft-voting ensemble of LightGBM (29 engineered
> URL features) and a character-level CNN, trained on N real URLs; achieved X% F1 / Y% recall on a
> held-out test set; FastAPI + ONNX backend with explained verdicts, Dockerised and deployed on Render.
