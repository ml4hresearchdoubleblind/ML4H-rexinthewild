#!/usr/bin/env python3
"""
Score model result CSVs against the benchmark gold answers.

Confidence intervals come from statsmodels
(`statsmodels.stats.proportion.proportion_confint`, method="wilson") -- the same
package the paper cites. Nothing here re-implements an interval by hand.

  python score_results.py results/gpt_960.csv results/gemini_960.csv

Blank predictions are counted as INCORRECT and reported separately. They are not
silently dropped from the denominator (which is what run_inference.py's
own calculate_accuracy does).
"""
import argparse, csv, os, string, sys
from collections import Counter, defaultdict
from statsmodels.stats.proportion import proportion_confint, proportions_chisquare

ALPHA = 0.05


def norm(s):
    s = (s or "").replace("•", "").strip()
    return "".join(s.lower().split()).translate(str.maketrans("", "", string.punctuation))


def ci(k, n):
    if n == 0:
        return (float("nan"), float("nan"))
    return proportion_confint(k, n, alpha=ALPHA, method="wilson")


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{path}: empty")
    pred_cols = [c for c in rows[0]
                 if c.endswith("_answer") and not c.endswith("_unformatted")]
    if len(pred_cols) != 1:
        sys.exit(f"{path}: expected exactly one *_answer column, found {pred_cols}")
    col = pred_cols[0]
    out = {}
    for x in rows:
        key = (x["pmcid"], x["image_file_name"], x["question"])
        pred = (x.get(col) or "").strip()
        out[key] = {"pred": pred, "gold": (x.get("answer") or "").strip(),
                    "correct": bool(pred) and norm(pred) == norm(x.get("answer")),
                    "blank": not bool(pred), "tag": x.get("tag", "")}
    return col, out


def report(name, d):
    n = len(d); k = sum(v["correct"] for v in d.values())
    blanks = sum(v["blank"] for v in d.values())
    lo, hi = ci(k, n)
    print(f"\n{name}   n={n}")
    print(f"  accuracy {k}/{n} = {k/n*100:.1f}%   95% CI [{lo*100:.1f}, {hi*100:.1f}]")
    if blanks:
        print(f"  blank predictions: {blanks} ({blanks/n*100:.1f}%) -- counted as incorrect")
    print(f"  {'category':<24} {'n':>4} {'acc':>7}   95% CI")
    by = defaultdict(list)
    for v in d.values():
        by[v["tag"]].append(v["correct"])
    for tag in sorted(by, key=lambda t: -len(by[t])):
        vals = by[tag]; kk = sum(vals); nn = len(vals); l, h = ci(kk, nn)
        print(f"  {tag:<24} {nn:>4} {kk/nn*100:6.1f}%   [{l*100:5.1f}, {h*100:5.1f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    a = ap.parse_args()

    loaded = {}
    for p in a.csvs:
        col, d = load(p)
        loaded[os.path.basename(p).replace(".csv", "")] = d
        report(f"{os.path.basename(p)}  [{col}]", d)

    if len(loaded) > 1:
        names = list(loaded)
        common = set.intersection(*(set(d) for d in loaded.values()))
        print(f"\n=== paired comparison on the {len(common)} questions all runs answered ===")
        for nm in names:
            d = loaded[nm]; k = sum(d[q]["correct"] for q in common); n = len(common)
            lo, hi = ci(k, n)
            print(f"  {nm:<16} {k}/{n} = {k/n*100:5.1f}%   [{lo*100:.1f}, {hi*100:.1f}]")
        if len(names) == 2:
            a_, b_ = (loaded[n] for n in names)
            # McNemar counts: disagreements only
            b01 = sum(1 for q in common if a_[q]["correct"] and not b_[q]["correct"])
            b10 = sum(1 for q in common if b_[q]["correct"] and not a_[q]["correct"])
            agree = sum(1 for q in common if a_[q]["correct"] == b_[q]["correct"])
            print(f"  agreement: {agree}/{len(common)} = {agree/len(common)*100:.1f}%")
            print(f"  {names[0]} right / {names[1]} wrong: {b01}")
            print(f"  {names[1]} right / {names[0]} wrong: {b10}")
            try:
                from statsmodels.stats.contingency_tables import mcnemar
                import numpy as np
                both = sum(1 for q in common if a_[q]["correct"] and b_[q]["correct"])
                neither = sum(1 for q in common if not a_[q]["correct"] and not b_[q]["correct"])
                res = mcnemar(np.array([[both, b01], [b10, neither]]), exact=True)
                print(f"  McNemar exact p = {res.pvalue:.4f}  (paired test on the disagreements)")
            except Exception as e:
                print(f"  (McNemar unavailable: {e})")


if __name__ == "__main__":
    main()
