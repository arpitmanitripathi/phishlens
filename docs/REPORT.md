# PhishLens — Project Report

*Living document, updated at the end of every phase.*
*Last updated: 2026-11-05 (Phases 0–1 complete; second bias audit).*

---

## 1. Introduction

### 1.1 Problem
Phishing is a cyber-attack in which a fake website imitates a trusted one (a bank, a delivery
company, a social network) to steal passwords or payment details. The victim usually reaches the
fake site by clicking a link in an e-mail, SMS or chat message. Detecting a phishing **URL**
before the page is opened is a cheap, fast first line of defence.

### 1.2 Objective
Build a real-time phishing URL detector that:
1. classifies a URL as **safe / suspicious / phishing** with a risk score (0–100),
2. explains **why** (top reasons),
3. works from the URL text alone (fast, and the server never visits a dangerous site),
4. is deployed as a live web application.

### 1.3 Approach (summary)
Supervised binary classification. Two different models look at the same URL in different ways
and their outputs are combined by **soft weighted voting**:
- **LightGBM** (gradient-boosted decision trees) on **29 hand-engineered features**,
- a **character-level CNN** (convolutional neural network) on the raw URL text.

### 1.4 Scope and constraints
- URL text only: no WHOIS, DNS or page-content checks (possible future work).
- Built as a fresher-level project: every component must be explainable in an interview.
- Free tooling and free hosting only.

---

## 2. Tools and environment (Phase 0)

| Area | Choice | Reason |
|---|---|---|
| Language | Python 3.12 | Widely supported by ML libraries (3.14 too new) |
| Environment | `uv` + virtual environment, locked versions (`uv.lock`) | Reproducible installs |
| Data | pandas, numpy, tldextract | Tabular data handling, correct domain splitting |
| ML | scikit-learn, LightGBM, PyTorch (training only), ONNX Runtime (serving) | Industry standard; small production image |
| Collection | requests, BeautifulSoup | Downloading feeds, parsing HTML for links |
| Analysis | matplotlib, Jupyter | EDA charts |
| Quality | pytest, ruff | Tests and code style |
| Hardware | Apple M5, 16 GB RAM (GPU via PyTorch MPS) | Sufficient; no cloud GPU needed |

Libraries are split into **runtime** (needed by the live app) and **dev** (training, crawling,
analysis) groups so the production image stays small enough for a free host.

---

## 3. Data collection (Phase 1)

### 3.1 Sources

| # | Source | Class | What it provides | Used |
|---|---|---|---|---|
| 1 | OpenPhish community feed | Phishing | ~300 live, verified phishing URLs | All (279 after cleaning/caps) |
| 2 | Phishing.Database (GitHub) | Phishing | 789,054 known active phishing URLs | Random capped sample (19,721) |
| 3 | Tranco top-1M list | Legit | Ranked list of popular **domain names** (no paths) | 7,000 sites chosen for crawling |
| 4 | Own crawl of Tranco sites | Legit | Homepage + internal links **with real paths** | 20,000 sampled |

### 3.2 Why a crawler was necessary
A domain list contains only names (`github.com`), while phishing URLs have long paths. Training on
that would let a model learn the shortcut *"has a path / is long ⇒ phishing"*. A public reference
project (PhishX) was found to have exactly this problem: **0 %** of its legit URLs had a path,
against 66 % of its phishing URLs. To avoid it, legit URLs were collected by visiting real sites.

### 3.3 Crawl method
- 7,000 sites: 3,000 random from Tranco ranks 1–10,000 (popular) + 4,000 random from ranks
  10,001–200,000 (smaller sites), so "legit" is not only big brands.
- Per site: `robots.txt` is checked first; then only the homepage is fetched (2 requests per
  site, never deeper). Up to 10 **internal** links are kept per site.
- The bot identifies itself with an honest User-Agent; 32 sites crawled in parallel; results
  are written incrementally and the crawl can resume after interruption.
- Result: **3,924 of 7,000 sites responded** (others blocked bots, had no homepage, or timed
  out); ~36,000 legit URLs collected in ~15 minutes.

### 3.4 Cleaning and sampling
1. **Normalisation** (`normalize_url`): trim whitespace, add a missing scheme, keep only
   http/https, lowercase scheme and host, remove default ports (`:443`, `:80`), reject
   `mailto:`/`javascript:`/invalid URLs.
2. **De-duplication.**
3. **Phishing sampling:** Phishing.Database is alphabetically sorted, so it is **shuffled**
   before sampling; caps of **3 URLs per host** and **200 per website** stop single campaigns or
   free hosts from dominating.
4. **Balancing:** 20,000 per class.
5. **Contradictions removed:** any URL present in both classes is dropped.
6. **`domain` column** added (registered domain, e.g. `login.paypal.co.uk → paypal.co.uk`) for
   splitting train/test by website later.
7. Fixed random seed (42) everywhere → the dataset is fully reproducible.

### 3.5 Final dataset

| Source | URLs | Label |
|---|---|---|
| Phishing.Database | 19,721 | phishing |
| OpenPhish | 279 | phishing |
| Crawled links | 17,836 | legit |
| Crawled homepages | 2,164 | legit |
| **Total** | **40,000** | 50 / 50 |

19,121 distinct websites (legit ≈ 3,600, phishing ≈ 15,500); only 8 websites contain both classes.

---

## 4. Exploratory data analysis (Phase 1)

Full notebook: `notebooks/eda.ipynb`. Charts: `docs/images/`.
All measurements use the model's view of a URL (no scheme, no `www.`).

### 4.1 Collection-bias check
A *collection bias* is a difference between classes caused by how the data was gathered rather
than by phishing itself. A model would learn it as a shortcut that fails in the real world.

| Property | Legit | Phishing | Assessment |
|---|---|---|---|
| Has a path or query | 87.6 % | 72.4 % | ✅ Shortcut removed by crawling |
| Median length | 32 chars | 43 chars | ✅ Real difference, distributions overlap heavily |
| Uses `http://` (not https) | 0.9 % | 23.8 % | ⚠️ Artefact: crawler started every site at https → hidden from models |
| Starts with `www.` | 51.0 % | 8.7 % | ⚠️ Artefact of site conventions → hidden from models |
| Default port `:443` | 2.4 % → **0 %** | 0.2 % | ⚠️ Artefact from redirects → fixed in normalisation |
| Ends with `/` | 39.4 % → **0.06 %** | 0.9 % → 0 % | 🚨 Artefact: feeds strip it, sites keep it → removed in normalisation |
| Non-ASCII characters | 0.6 % → **0 %** | 0 % | ⚠️ Artefact: feeds store encoded URLs → %-encoding / punycode in normalisation |
| `#fragment` | 0 % | 0.04 % → 0 % | ⚠️ Artefact: crawler dropped it → dropped everywhere; feature `has_fragment` replaced by `is_homepage` |
| Free hosting / site builders | 0.1 % | 22.4 % | 🚨 Partly artefact: Tranco lists only main domains, never user sites (`someone.weebly.com`) → see §6.2 |
| Has `?query` | 7.2 % | 18.5 % | 🟡 Partly artefact (homepage navigation links rarely have queries) → documented, monitored |
| `.php` / `.html` path | 10.0 % | 33.0 % | ✅ Real (phishing kits are mostly PHP) |
| Bare homepage | 12.4 % | 27.6 % | ✅ Real (many phishing sites live at the root) |

### 4.2 Findings

| Finding | Numbers | Use in features |
|---|---|---|
| Suspicious words far more frequent in phishing | `login` 10.4 % vs 0.7 %; `verify`, `confirm`, `wallet` ≈ 0 % in legit | `num_suspicious_words` |
| Many more digits | mean 10.5 vs 2.0 per URL | `num_digits`, `digit_ratio_in_domain` |
| More subdomains | mean 0.7 vs 0.1 | `num_subdomains` |
| Cheap / new TLDs lean phishing | `.dev .app .io .top .xyz` | `suspicious_tld` |
| Brand impersonation | 14.7 % of phishing vs 1.1 % of legit mention a brand on a non-brand site | `brand_in_url_not_domain`, `brand_lookalike` |
| Length overlaps, phishing long tail | 4.9 % of phishing > 200 chars | `url_length`, `path_length` |
| **Legit has more hyphens** (path slugs like `/pasta-recipes`) | 1.7 vs 1.1 | changed to `num_hyphens_in_domain` |
| Rare but one-sided flags | IP host 1.2 % vs 0 %; `@` 2.0 % vs 0.03 % | `has_ip`, `has_at` |
| Weak signals | punycode, %-encoding similar in both classes | kept; model decides |
| Phishing relies on free hosting | 000webhostapp, weebly, gitbook, godaddysites | known limitation |

---

## 5. Design decisions log

| # | Decision | Reason |
|---|---|---|
| D1 | URL-text-only features | Fast (ms), safe (no visiting malicious pages), simple |
| D2 | Crawl legit URLs with paths | Avoid the length/path shortcut seen in PhishX |
| D3 | Hide scheme (`http/https`) and `www.` from models | Both are collection artefacts in this dataset (§4.1, §6.1) |
| D4 | Strip default ports in normalisation | `:443` appeared only in legit URLs because of redirects |
| D5 | Caps per host/website when sampling phishing | Variety; prevent single campaigns dominating |
| D6 | `domain` column; split train/test by website (Phase 3) | Prevent the model "recognising" a site instead of judging the URL |
| D7 | Count hyphens in the domain only | EDA: legit paths use more hyphens than phishing |
| D8 | Shared `urls.py` used by dataset, features and app | No train/serve mismatch (the bug in the original project) |
| D10 | Normalise trailing `/`, `#fragment`, non-ASCII for every URL | Second bias audit found each to be a collection artefact (§4.1) |
| D9 | LightGBM + Char-CNN, soft voting; LR/DT/RF only as benchmark | Two genuinely different views of a URL; benchmark justifies the choice |

---

## 6. Discussion notes

### 6.1 Why the models do not see `http://` / `https://` / `www.`
In the real world, plain `http://` is a *weak* warning sign, but in **our dataset** it is mostly
an artefact: every legit URL was collected starting from `https://`, so only 0.9 % of legit URLs
are http versus 23.8 % of phishing. A model shown the scheme would learn
*"http ⇒ almost certainly phishing"* far more strongly than reality justifies, flagging small
legitimate http sites and under-weighting real clues for https phishing (76 % of our phishing
URLs already use https). The scheme can still be shown to the user as a separate, rule-based
note (*"connection is not encrypted"*) without affecting the model score.

### 6.2 Free hosting imbalance
Attackers genuinely favour free hosting and site builders (000webhostapp, weebly, github.io,
vercel.app…), but millions of legitimate sites also live there. Tranco ranks main domains only,
so almost no legit user sites on these platforms entered the dataset (0.1 % vs 22.4 %). A model
trained like this would treat *every* site on these platforms as phishing. Status: open decision.

---

## 7. Limitations (so far)
- URL-only detection cannot catch phishing hosted on compromised legitimate sites or free hosts
  whose URLs look clean.
- Legit links come from homepages only (often navigation pages), slightly cleaner than deep pages.
- Phishing feeds contain some noise (dead or mislabelled URLs).
- Scores measured on URL datasets tend to overstate real-world performance.

---

## 8. Upcoming phases
2. Feature extraction (29 features) and character encoding
3. Benchmark, LightGBM, Char-CNN, voting, evaluation
4. Explanations
5. FastAPI application and web page
6. Testing, Docker, deployment
