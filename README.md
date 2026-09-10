# ReXInTheWild

A benchmark of clinician-verified multiple-choice questions over **patient-style
medical photographs** — the kind of images people actually capture with a phone,
rather than the standardised radiographs and dermoscopy images most medical VQA
benchmarks are built from.

**942 questions** over **479 images** drawn from **379 open-access PubMed Central
articles**, each question reviewed by clinicians across two rounds.

This repository accompanies a paper under review at ML4H 2026. It is an anonymous
release for the review period.

## Contents

| Path | What it is |
|---|---|
| `rexinthewild.csv` | The benchmark: 942 questions, answer keys, topic tags, and source attribution |
| `fetch_images.py` | Retrieves the images from PMC Open Access (see below) |
| `prompts.md` | Every pipeline prompt and all generation/evaluation hyperparameters |
| `source_articles.csv` | Per-article licence, DOI, citation, and image/question counts for all 379 sources |
| `code/` | Inference, scoring and fine-tuning code — see `code/README.md` |
| `code/run_inference.py` | Evaluation harness for the models reported in the paper |
| `code/run_with_provenance.py` | Wrapper recording model IDs, decoding params, and checksums per run |
| `code/score_results.py` | Scores result CSVs with Wilson confidence intervals |
| `code/train_cv_qwen8b.py` | QLoRA fine-tuning with cross-validation (paper appendix) |
| `code/cv_folds.json` | The exact folds used, split by image so no image spans two folds |
| `ATTRIBUTION.md` | Licensing, image provenance, and how to cite the source articles |

## Images are not redistributed

The CSV contains questions only. Images stay with their source articles, under
their own licences, and are fetched on demand:

```bash
python fetch_images.py rexinthewild.csv images/
```

No dependencies beyond the standard library, and nothing to configure. Each file
is written to exactly the path its `file_name` column names, so after fetching
into `images/` the dataset's paths resolve as-is. Images are pulled as individual
objects from the PMC Cloud Service on AWS Open Data — no bulk index, no tarballs.
All 479 were verified retrievable at the time of release.

> **Licensing, in short: every one of the 379 source articles is non-commercial
> only, and about a third are additionally no-derivatives.** Not one is plain
> CC BY. Read `ATTRIBUTION.md` before using the images for anything, and consult
> `source_articles.csv` for per-article terms.

If you built anything against NCBI's old FTP endpoints, note that they were
**retired in August 2026** — `oa_file_list.csv` and `oa.fcgi` both 404 now. See
the end of `ATTRIBUTION.md` for the current object layout.

## Schema

| Column | Description |
|---|---|
| `file_name` | Relative image path, `images/{PMCID}_{basename}`. Unique per image. |
| `question` | Question stem |
| `choice_a` … `choice_e` | Answer options as separate columns. Populated in order; unused ones are empty. |
| `answer` | **The full text of the correct option**, not a letter |
| `tag` | One of 7 body-region topics |
| `pmc_url` | Source article on PMC |
| `title` | Source article title |
| `authors` | Source article authors — **required for licence attribution**, see `ATTRIBUTION.md` |

`answer` holds option *text* rather than a letter, so scoring does not depend on
option order and you can shuffle options freely without remapping a key. Match
the model's chosen option text against `answer`:

```python
import pandas as pd
d = pd.read_csv("rexinthewild.csv")
CH = ["choice_a", "choice_b", "choice_c", "choice_d", "choice_e"]
d["options"] = d[CH].apply(lambda r: [x for x in r if isinstance(x, str) and x], axis=1)
assert all(a in o for a, o in zip(d.answer, d.options))   # holds for all 942
```

The `PMCID_` prefix in `file_name` is what makes it unique. Basenames alone are
not: publishers reuse generic figure names, so `gr1.jpg`, `gr2.jpg`, and `gr4.jpg`
each appear in several different articles. There are 479 distinct images but only
471 distinct basenames.

## Evaluation notes

Questions have **3, 4, or 5 options** (4 / 451 / 487 questions respectively), so
the chance baseline is **22.4%**, not 25%. Report it that way; assuming a uniform
option count overstates how far above chance a model is.

Topic tags are unbalanced by construction, reflecting what is actually
photographed and published:

| Tag | n |
|---|---|
| Trunk & Extremities | 369 |
| Head & Neck | 245 |
| Eyes | 148 |
| Mouth & Jaws | 110 |
| Skin & Hair | 35 |
| Surgical & Procedural | 25 |
| Other | 10 |

Per-tag accuracies on the smaller tags carry wide confidence intervals. The paper
reports Wilson intervals throughout for this reason.

## Reproducing the paper's numbers

`code/run_inference.py` covers the evaluated models behind a single interface.
API credentials are read from the environment, never from files in this repo:

```bash
python code/run_with_provenance.py --model <key> --dataset rexinthewild.csv --outdir results/
```

`run_with_provenance.py` writes a `.meta.json` alongside each result file
recording the resolved model ID, every decoding parameter, the served model
snapshot where the provider exposes it, and SHA-256 checksums of both the dataset
and the scripts. Closed-model results are not bitwise reproducible — providers
update served checkpoints — so those snapshot fields are what let you tell
whether a divergence is yours or theirs.

For the fine-tuning appendix, `code/train_cv_qwen8b.py` uses the folds in
`cv_folds.json`. Folds are split by image, so no image appears in both a training
and a validation fold, and are stratified by tag.

## Licence

Code: MIT (`LICENSE`). Questions and answer keys: CC BY 4.0. Images: **not
redistributed here**, and each is governed by its source article's own licence,
all of which are non-commercial. See `ATTRIBUTION.md`.

## Citation

Citation details will be added once review concludes.
