#!/usr/bin/env python3
"""
Fetch the ReXInTheWild images from the PMC Open Access Subset.

The benchmark ships question-answer pairs only. Images are not redistributed;
this script retrieves each one from its source article so that you obtain it
under that article's own licence. Every source article is non-commercially
licensed -- see ATTRIBUTION.md before using the images for anything.

Images come from the PMC Cloud Service on AWS Open Data, which replaced NCBI's
legacy FTP dataset service in August 2026. Objects are addressed directly, so
there is no bulk index to download and no tarballs to unpack.

The `file_name` column already holds the relative path each image is written to
(images/{PMCID}_{basename}), so pointing --out at `images/` reproduces the paths
the dataset refers to. The PMCID prefix is what makes the name unique: generic
basenames like gr1.jpg recur across articles.

Usage:
    python fetch_images.py rexinthewild.csv images/
    python fetch_images.py rexinthewild.csv images/ --workers 8
"""
import argparse
import csv
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com"
UA = "rexinthewild-fetch/1.0 (https://github.com/ml4hresearchdoubleblind/ML4H-rexinthewild)"


def http_get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def resolve_version(pmcid, cache={}):
    """PMC objects are keyed PMC<id>.<version>/. Find the version that exists."""
    if pmcid in cache:
        return cache[pmcid]
    for v in (1, 2, 3):
        try:
            urllib.request.urlopen(
                urllib.request.Request(
                    f"{BUCKET}/{pmcid}.{v}/{pmcid}.{v}.json",
                    headers={"User-Agent": UA}, method="HEAD"), timeout=30)
            cache[pmcid] = v
            return v
        except urllib.error.HTTPError:
            continue
        except Exception:
            break
    try:  # fall back to a listing
        xml = http_get(f"{BUCKET}/?list-type=2&prefix={pmcid}.").decode("utf-8", "replace")
        for key in re.findall(r"<Key>([^<]+)</Key>", xml):
            if key.endswith(".json"):
                v = int(key.split("/")[0].rsplit(".", 1)[1])
                cache[pmcid] = v
                return v
    except Exception:
        pass
    cache[pmcid] = None
    return None


def fetch_one(task, out_dir):
    pmcid, basename, version = task
    dest = out_dir / f"{pmcid}_{basename}"
    if dest.exists() and dest.stat().st_size > 0:
        return ("skip", pmcid, basename, "already present")
    if version is None:
        version = resolve_version(pmcid)
    if version is None:
        return ("fail", pmcid, basename, "no PMC object found for this article")
    try:
        data = http_get(f"{BUCKET}/{pmcid}.{version}/{basename}")
    except urllib.error.HTTPError as e:
        return ("fail", pmcid, basename, f"HTTP {e.code}")
    except Exception as e:
        return ("fail", pmcid, basename, str(e)[:80])
    if not data:
        return ("fail", pmcid, basename, "empty response")
    dest.write_bytes(data)
    return ("ok", pmcid, basename, f"{len(data)} bytes")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path", help="rexinthewild.csv")
    ap.add_argument("out_dir", help="directory to write images into")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel downloads (default 8; please keep this modest)")
    ap.add_argument("--articles", default=None,
                    help="source_articles.csv, to skip per-article version lookups")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    versions = {}
    articles = args.articles
    if articles is None:
        default = Path(args.csv_path).with_name("source_articles.csv")
        articles = str(default) if default.exists() else None
    if articles:
        with open(articles, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    versions[row["pmcid"]] = int(row["pmc_version"])
                except (KeyError, ValueError):
                    pass
        print(f"[info] read {len(versions)} article versions from {articles}")

    seen, tasks, unparsed = set(), [], 0
    with open(args.csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            # file_name is "images/PMC1234567_figure.jpg"
            m = re.match(r"^(?:.*/)?(PMC\d+)_(.+)$", (row.get("file_name") or "").strip())
            if not m:
                unparsed += 1
                continue
            pmcid, base = m.group(1), m.group(2)
            if (pmcid, base) in seen:
                continue
            seen.add((pmcid, base))
            tasks.append((pmcid, base, versions.get(pmcid)))
    if unparsed:
        print(f"[warn] {unparsed} rows had an unparseable file_name and were skipped")

    print(f"[info] {len(tasks)} distinct images across "
          f"{len({t[0] for t in tasks})} articles")

    counts, failures = Counter(), []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (status, pmcid, base, note) in enumerate(
                ex.map(lambda t: fetch_one(t, out_dir), tasks), 1):
            counts[status] += 1
            if status == "fail":
                failures.append((pmcid, base, note))
                print(f"[fail] {pmcid} {base}: {note}")
            if i % 50 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}  ok={counts['ok']} "
                      f"skip={counts['skip']} fail={counts['fail']}", flush=True)

    print(f"\ndone. retrieved {counts['ok']}, already present {counts['skip']}, "
          f"failed {counts['fail']}")
    if failures:
        print("\nArticles are occasionally re-versioned or withdrawn upstream. Re-run to "
              "retry; if a failure persists, please open an issue with the pmcid and filename.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
