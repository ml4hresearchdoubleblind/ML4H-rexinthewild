# ReXInTheWild

A benchmark of clinician-verified multiple-choice questions over **patient-style
medical photographs** — the kind of images people actually capture with a phone,
rather than the standardised radiographs and dermoscopy images that most medical
VQA benchmarks are built from.

**954 questions** over **485 images** drawn from **381 open-access PubMed Central
articles**, each question reviewed by clinicians across two rounds.

This repository accompanies a paper under review at ML4H 2026. It is an anonymous
release for the review period.

## Contents

| Path | What it is |
|---|---|
| `rexinthewild.csv` | The benchmark: 954 questions with answer keys and topic tags |
| `fetch_images.py` | Retrieves the images from PMC Open Access (see below) |
| `code/run_inference.py` | Evaluation harness for the models reported in the paper |
| `code/run_with_provenance.py` | Wrapper recording model IDs, decoding params, and checksums per run |
| `code/train_cv_qwen8b.py` | QLoRA fine-tuning with 5-fold cross-validation (paper appendix) |
| `code/cv_folds.json` | The exact folds used, split by image so no image spans two folds |
| `ATTRIBUTION.md` | Licensing, image provenance, and how to cite the source articles |

## Images are not redistributed

The CSV contains questions only. Images stay with their source articles, under
their own licences, and are fetched on demand:

```bash
pip install requests
python fetch_images.py rexinthewild.csv images/ --email you@example.com
```

NCBI asks that automated requests carry a contact address, which is why
`--email` is required. The script groups requests by article, so each OA package
is downloaded once even when several questions share a figure, and it caches the
OA file index between runs. Expect the first run to take a while — it downloads
NCBI's full OA index before fetching anything.

See `ATTRIBUTION.md` for licence terms. Not every article carries the same
licence, and a few permit non-commercial use only.

## Schema

| Column | Description |
|---|---|
| `image_id` | **Unique image key**: `{pmcid}_{image_file_name}`. Matches the filename written by `fetch_images.py`. |
| `image_file_name` | Filename within the source article. **Not unique on its own** — see below. |
| `pmcid` | PMC accession of the source article |
| `question` | Question stem |
| `choices` | Answer options, one per line, each formatted `[X] option text` |
| `answer` | Gold option letter |
| `answer_text` | Text of the gold option (redundant with `answer`, provided for convenience) |
| `tag` | One of 7 body-region topics |

**Join on `image_id`, not `image_file_name`.** Publishers reuse generic figure
names, so `gr1.jpg`, `gr2.jpg`, and `gr4.jpg` each appear in several different
articles. There are 485 distinct images but only 476 distinct filenames; keying
on the filename silently collides 9 of them.

`choices` is a newline-delimited string, not a serialised list. To parse:

```python
import re, pandas as pd
d = pd.read_csv("rexinthewild.csv")
options = d.choices.map(lambda s: dict(re.findall(r"^\[([A-Z])\]\s*(.*?)\s*$", s, re.M)))
```

## Evaluation notes

Questions have **3, 4, or 5 options** (4 / 454 / 496 questions respectively), so
the chance baseline is **22.4%**, not 25%. Report it that way; assuming a uniform
option count overstates how far above chance a model is.

Option order in the CSV is fixed and deterministic. If you shuffle options to
control for position bias, seed the shuffle per question so results stay
reproducible — the harness in `code/` derives its seed from a hash of the
question text for this reason.

Topic tags are unbalanced by construction, reflecting what is actually
photographed and published:

| Tag | n |
|---|---|
| Trunk & Extremities | 371 |
| Head & Neck | 245 |
| Eyes | 150 |
| Mouth & Jaws | 115 |
| Skin & Hair | 37 |
| Surgical & Procedural | 26 |
| Other | 10 |

Per-tag accuracies on the smaller tags carry wide confidence intervals. The paper
reports Wilson intervals throughout for this reason.

## Reproducing the paper's numbers

`code/run_inference.py` covers the evaluated models behind a single interface.
API credentials are read from the environment and never read from files in this
repository:

```bash
python code/run_with_provenance.py --model <key> --dataset rexinthewild.csv --outdir results/
```

`run_with_provenance.py` writes a `.meta.json` alongside each result file
recording the resolved model ID, every decoding parameter, the served model
snapshot where the provider exposes it, and SHA-256 checksums of both the dataset
and the scripts. Closed-model results are not bitwise reproducible — providers
update served checkpoints — so the snapshot fields are what let you tell whether
a divergence is yours or theirs.

For the fine-tuning appendix, `code/train_cv_qwen8b.py` uses the folds in
`cv_folds.json`. Folds are split by image, so no image appears in both a training
and a validation fold, and are stratified by tag.

## Licence

Code in this repository: MIT. See `ATTRIBUTION.md` for the questions and images,
which carry different terms.

## Citation

Citation details will be added once review concludes.
