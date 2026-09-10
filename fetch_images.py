#!/usr/bin/env python3
"""
Fetch the ReXInTheWild images from the PubMed Central Open Access subset.

The benchmark ships question-answer pairs only. Images are not redistributed;
this script retrieves each one from its source article's PMC OA package, so
the figures you evaluate on are the originals under their own licences.

Images are grouped by article, so each OA package is downloaded once even
when several questions reference the same figure (954 questions span 381
articles). Files are written as {pmcid}_{image_file_name}, which matches the
`image_id` column and is the dataset's unique image key -- `image_file_name`
alone is NOT unique, since generic names like gr1.jpg recur across articles.

Usage:
    python fetch_images.py rexinthewild.csv images/ --email you@example.com

NCBI asks that automated requests identify a contact address; --email is
required for that reason. Please also keep --sleep at or above 0.34s to stay
within the 3 requests/second guidance.
"""
import argparse
import csv
import io
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

import requests

OA_FILE_LIST = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_file_list.csv"
FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/"


def get(url, email, timeout=120, stream=False):
    headers = {"User-Agent": f"rexinthewild-fetch/1.0 (contact: {email})"}
    r = requests.get(url, headers=headers, timeout=timeout, stream=stream)
    r.raise_for_status()
    return r


def load_oa_index(outdir, email, wanted):
    """Map PMCID -> relative tar path, for the PMCIDs we actually need."""
    cache = outdir / "oa_file_list.csv"
    if not cache.exists():
        print(f"[info] downloading OA file list (~1GB uncompressed) -> {cache}")
        r = get(OA_FILE_LIST, email, timeout=600, stream=True)
        with cache.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    else:
        print(f"[info] using cached OA file list at {cache}")

    index = {}
    with cache.open(newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            acc = row.get("Accession ID")
            if acc in wanted:
                index[acc] = row.get("File")
    missing = wanted - set(index)
    if missing:
        print(f"[warn] {len(missing)} PMCIDs absent from the OA list: "
              f"{sorted(missing)[:5]}{' ...' if len(missing) > 5 else ''}")
    return index


def extract_from_package(tar_bytes, basenames):
    """Pull the requested basenames out of one OA package tarball."""
    found = {}
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as t:
        members = [m for m in t.getmembers() if m.isfile()]
        by_base = {m.name.rsplit("/", 1)[-1].lower(): m for m in members}
        for base in basenames:
            m = by_base.get(base.lower())
            if m is None:
                continue
            fh = t.extractfile(m)
            if fh is not None:
                found[base] = fh.read()
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path", help="rexinthewild.csv")
    ap.add_argument("out_dir", help="directory to write images into")
    ap.add_argument("--email", required=True,
                    help="contact address sent to NCBI, per their access policy")
    ap.add_argument("--sleep", type=float, default=0.34,
                    help="seconds between requests (default 0.34, ~3/s)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # group the wanted basenames by article
    by_pmcid = defaultdict(set)
    with open(args.csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pmcid = (row.get("pmcid") or "").strip()
            name = (row.get("image_file_name") or "").strip()
            if pmcid and name:
                by_pmcid[pmcid].add(name)
    n_img = sum(len(v) for v in by_pmcid.values())
    print(f"[info] {n_img} images across {len(by_pmcid)} articles")

    index = load_oa_index(out, args.email, set(by_pmcid))

    ok = skipped = failed = 0
    for i, (pmcid, basenames) in enumerate(sorted(by_pmcid.items()), 1):
        todo = {b for b in basenames if not (out / f"{pmcid}_{b}").exists()}
        skipped += len(basenames) - len(todo)
        if not todo:
            continue

        rel = index.get(pmcid)
        if not rel:
            print(f"[error] {pmcid}: no OA package entry")
            failed += len(todo)
            continue

        try:
            r = get(FTP_BASE + rel, args.email, timeout=300)
            found = extract_from_package(r.content, todo)
        except Exception as exc:
            print(f"[error] {pmcid}: {exc}")
            failed += len(todo)
            continue

        for base in sorted(todo):
            data = found.get(base)
            if data is None:
                print(f"[error] {pmcid}: {base} not in package {rel}")
                failed += 1
                continue
            (out / f"{pmcid}_{base}").write_bytes(data)
            ok += 1

        print(f"[{i}/{len(by_pmcid)}] {pmcid}: {len(found)}/{len(todo)} extracted")
        time.sleep(args.sleep)

    print(f"\ndone. retrieved {ok}, already present {skipped}, failed {failed}")
    if failed:
        print("Articles can be withdrawn or re-versioned upstream; re-run to retry, "
              "and see ATTRIBUTION.md for how to report persistent gaps.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
