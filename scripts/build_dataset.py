"""Build the PhishLens dataset: data/processed/dataset.csv  (url, label, source, domain).

Three steps, each can be run on its own:

    uv run python scripts/build_dataset.py download   # fetch phishing feeds + Tranco list
    uv run python scripts/build_dataset.py crawl      # collect legit URLs with real paths
    uv run python scripts/build_dataset.py build      # clean, sample, balance, save
    uv run python scripts/build_dataset.py all        # everything in order

label: 1 = phishing, 0 = legit
"""

import argparse
import csv
import random
import sys
import threading
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
from bs4 import BeautifulSoup

from phishlens.urls import hostname, normalize_url, registered_domain

# ---------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

SEED = 42
TARGET_PER_CLASS = 20_000

SOURCES = {
    "openphish.txt": "https://openphish.com/feed.txt",
    "phishing_database_active.txt": (
        "https://raw.githubusercontent.com/Phishing-Database/Phishing.Database/"
        "master/phishing-links-ACTIVE.txt"
    ),
    "tranco.zip": "https://tranco-list.eu/top-1m.csv.zip",
}

# Phishing sampling caps: stop a few big hosts / campaigns from dominating the data.
MAX_PHISH_PER_HOST = 3  # e.g. at most 3 URLs from "abc.000webhostapp.com"
MAX_PHISH_PER_SITE = 200  # e.g. at most 200 URLs from all of "000webhostapp.com"

# Crawl settings: which Tranco sites to visit and how politely.
CRAWL_TOP_SITES = 3_000  # random sites from Tranco ranks 1 - 10,000 (popular)
CRAWL_LONG_TAIL_SITES = 4_000  # random sites from ranks 10,001 - 200,000 (smaller sites)
MAX_LINKS_PER_SITE = 10  # internal links kept per site, so no site dominates
CRAWL_WORKERS = 32  # sites crawled in parallel (each site still gets only ~2 requests)
TIMEOUT = (5, 10)  # (connect, read) seconds
MAX_HTML_BYTES = 2_000_000
USER_AGENT = (
    "PhishLensDatasetBot/0.1 (+https://github.com/arpitmanitripathi/phishlens; "
    "academic project, homepage-only crawl)"
)

CRAWL_OUT = RAW / "legit_crawled.csv"  # url, site, kind (homepage / link)
CRAWL_DONE = RAW / "legit_crawled_done.txt"  # sites already attempted (lets us resume)


# ---------------------------------------------------------------------------------------------
# Step 1: download
# ---------------------------------------------------------------------------------------------
def download() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for filename, url in SOURCES.items():
        print(f"Downloading {url}")
        resp = requests.get(url, timeout=120, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        (RAW / filename).write_bytes(resp.content)
        print(f"  saved {filename} ({len(resp.content) / 1e6:.1f} MB)")

    with zipfile.ZipFile(RAW / "tranco.zip") as z:
        z.extractall(RAW)  # -> top-1m.csv
    print("Extracted top-1m.csv")


# ---------------------------------------------------------------------------------------------
# Step 2: crawl legit URLs
# ---------------------------------------------------------------------------------------------
def load_tranco() -> list[str]:
    with open(RAW / "top-1m.csv", newline="") as f:
        return [row[1] for row in csv.reader(f)]  # rows are: rank,domain (already in order)


def choose_sites_to_crawl() -> list[str]:
    tranco = load_tranco()
    rng = random.Random(SEED)
    top = rng.sample(tranco[:10_000], CRAWL_TOP_SITES)
    tail = rng.sample(tranco[10_000:200_000], CRAWL_LONG_TAIL_SITES)
    return top + tail


def crawl_site(session: requests.Session, site: str) -> list[tuple[str, str]]:
    """Visit ONE site's homepage and return [(url, kind), ...].

    Only two requests per site: robots.txt + the homepage. Never follows links further.
    """
    base = f"https://{site}/"

    # Respect robots.txt: skip the site if it disallows our bot on the homepage.
    robots = RobotFileParser()
    try:
        r = session.get(urljoin(base, "/robots.txt"), timeout=TIMEOUT)
        if r.status_code == 200:
            robots.parse(r.text.splitlines())
            if not robots.can_fetch(USER_AGENT, base):
                return []
    except requests.RequestException:
        pass  # no robots.txt reachable -> allowed by convention

    try:
        resp = session.get(base, timeout=TIMEOUT, stream=True)
        if resp.status_code != 200 or "text/html" not in resp.headers.get("Content-Type", ""):
            return []
        html = resp.raw.read(MAX_HTML_BYTES, decode_content=True).decode(
            resp.encoding or "utf-8", errors="replace"
        )
        final_url = resp.url  # after redirects, e.g. https://www.site.com/en/
    except (requests.RequestException, OSError, LookupError):
        return []

    homepage = normalize_url(final_url)
    if not homepage:
        return []
    site_domain = registered_domain(homepage)

    links = set()
    for a in BeautifulSoup(html, "html.parser").find_all("a", href=True):
        url = normalize_url(urljoin(final_url, a["href"]))
        if not url or registered_domain(url) != site_domain:
            continue  # skip invalid URLs and links to OTHER websites
        url = url.split("#")[0]  # "#section" links are the same page
        if url.rstrip("/") != homepage.rstrip("/"):
            links.add(url)

    rng = random.Random(f"{SEED}-{site}")
    chosen = rng.sample(sorted(links), min(MAX_LINKS_PER_SITE, len(links)))
    return [(homepage, "homepage")] + [(u, "link") for u in chosen]


def crawl() -> None:
    sites = choose_sites_to_crawl()
    done = set(CRAWL_DONE.read_text().split()) if CRAWL_DONE.exists() else set()
    todo = [s for s in sites if s not in done]
    print(f"Sites: {len(sites)} total, {len(done)} already done, {len(todo)} to crawl")

    new_file = not CRAWL_OUT.exists()
    lock = threading.Lock()
    thread_local = threading.local()

    def get_session() -> requests.Session:
        if not hasattr(thread_local, "session"):
            s = requests.Session()
            s.headers["User-Agent"] = USER_AGENT
            thread_local.session = s
        return thread_local.session

    started = time.time()
    with (
        open(CRAWL_OUT, "a", newline="") as out_f,
        open(CRAWL_DONE, "a") as done_f,
        ThreadPoolExecutor(CRAWL_WORKERS) as pool,
    ):
        writer = csv.writer(out_f)
        if new_file:
            writer.writerow(["url", "site", "kind"])
        futures = {pool.submit(crawl_site, get_session(), s): s for s in todo}
        ok = 0
        for i, fut in enumerate(as_completed(futures), 1):
            site = futures[fut]
            try:
                rows = fut.result()
            except Exception:  # one broken site must never stop the whole crawl
                rows = []
            with lock:  # results are written as they arrive, so a crash loses nothing
                writer.writerows((url, site, kind) for url, kind in rows)
                done_f.write(site + "\n")
                ok += bool(rows)
            if i % 250 == 0 or i == len(todo):
                out_f.flush()
                done_f.flush()
                print(f"  {i}/{len(todo)} sites, {ok} responded, {time.time() - started:.0f}s")


# ---------------------------------------------------------------------------------------------
# Step 3: build the final dataset
# ---------------------------------------------------------------------------------------------
def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def clean(urls: list[str]) -> list[str]:
    """Normalise, drop invalid, remove duplicates (keeping first-seen order)."""
    seen, out = set(), []
    for raw in urls:
        url = normalize_url(raw)
        if url and url not in seen:
            seen.add(url)
            out.append(url)
    return out


def sample_phishing(openphish: list[str], phishdb: list[str], n: int) -> list[tuple[str, str]]:
    """All OpenPhish URLs (freshest) + a random, capped sample of Phishing.Database."""
    rng = random.Random(SEED)
    phishdb = phishdb[:]
    rng.shuffle(phishdb)  # the file is alphabetical, so a random order is essential

    per_host, per_site = Counter(), Counter()
    chosen = []
    for url, source in [(u, "openphish") for u in openphish] + [
        (u, "phishing_database") for u in phishdb
    ]:
        host, site = hostname(url), registered_domain(url)
        if per_host[host] >= MAX_PHISH_PER_HOST or per_site[site] >= MAX_PHISH_PER_SITE:
            continue
        per_host[host] += 1
        per_site[site] += 1
        chosen.append((url, source))
        if len(chosen) == n:
            break
    return chosen


def sample_legit(crawled: pd.DataFrame, n: int) -> list[tuple[str, str]]:
    """All crawled homepages + links, down-sampled to n if we have more than enough."""
    rows = list(zip(crawled["url"], "tranco_" + crawled["kind"], strict=True))
    if len(rows) > n:
        rows = random.Random(SEED).sample(rows, n)
    return rows


def build() -> None:
    print("Cleaning phishing sources...")
    openphish = clean(read_lines(RAW / "openphish.txt"))
    phishdb = clean(read_lines(RAW / "phishing_database_active.txt"))
    print(f"  OpenPhish: {len(openphish):,} valid | Phishing.Database: {len(phishdb):,} valid")

    print("Cleaning crawled legit URLs...")
    crawled = pd.read_csv(CRAWL_OUT)
    crawled["url"] = crawled["url"].map(normalize_url)
    crawled = crawled.dropna(subset=["url"]).drop_duplicates(subset="url")
    print(f"  Crawled legit: {len(crawled):,} valid ({(crawled['kind'] == 'link').sum():,} links)")

    phishing = sample_phishing(openphish, phishdb, TARGET_PER_CLASS)
    legit = sample_legit(crawled, TARGET_PER_CLASS)

    # Balance: both classes get the same size (the smaller of the two).
    n = min(len(phishing), len(legit))
    if n < TARGET_PER_CLASS:
        print(f"  NOTE: only {n:,} per class available (target {TARGET_PER_CLASS:,})")
    rng = random.Random(SEED)
    phishing = rng.sample(phishing, n)
    legit = rng.sample(legit, n)

    df = pd.DataFrame(
        [(u, 1, s) for u, s in phishing] + [(u, 0, s) for u, s in legit],
        columns=["url", "label", "source"],
    )

    # A URL in BOTH classes would be a contradiction -> remove it entirely.
    conflicted = df["url"].duplicated(keep=False)
    if conflicted.any():
        print(f"  Removed {conflicted.sum()} URLs that appeared in both classes")
        df = df[~conflicted]

    df["domain"] = df["url"].map(registered_domain)  # used later to split by website
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle rows

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "dataset.csv"
    df.to_csv(out, index=False)
    report(df, out)


def report(df: pd.DataFrame, out: Path) -> None:
    print(f"\nSaved {out.relative_to(ROOT)}: {len(df):,} rows")
    print(df["label"].map({1: "phishing", 0: "legit"}).value_counts().to_string())
    print("\nBy source:")
    print(df["source"].value_counts().to_string())
    print(f"\nDistinct websites (domains): {df['domain'].nunique():,}")
    by_domain = defaultdict(set)
    for d, label in zip(df["domain"], df["label"], strict=True):
        by_domain[d].add(label)
    both = sum(1 for labels in by_domain.values() if len(labels) == 2)
    print(f"Domains with BOTH phishing and legit URLs: {both}")


# ---------------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("step", choices=["download", "crawl", "build", "all"])
    step = parser.parse_args().step

    if step in ("download", "all"):
        download()
    if step in ("crawl", "all"):
        crawl()
    if step in ("build", "all"):
        build()


if __name__ == "__main__":
    sys.exit(main())
