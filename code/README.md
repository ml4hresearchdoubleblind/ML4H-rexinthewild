# Inference and evaluation code

Everything needed to reproduce the model results in the paper. All prompts and
decoding settings are documented in `../prompts.md`; this directory is the code
that applies them.

| File | Purpose |
|---|---|
| `run_inference.py` | Runs one model over the benchmark. Covers every evaluated model behind a single interface. |
| `run_with_provenance.py` | Wrapper around the above that records exactly what ran, for reproducibility. |
| `score_results.py` | Scores result CSVs against the gold answers, with Wilson confidence intervals. |
| `train_cv_qwen8b.py` | QLoRA fine-tuning with cross-validation (paper appendix). |
| `cv_folds.json` | The exact folds used, split by image so no image spans two folds. |

## Running a model

Credentials are read from the environment and never from files in this
repository. Set whichever the target model needs:

```bash
export AZURE_OPENAI_API_KEY=...       AZURE_OPENAI_ENDPOINT=...
export ANTHROPIC_API_KEY=...
export GOOGLE_APPLICATION_CREDENTIALS=...   # Vertex AI service account JSON
```

Then:

```bash
python run_with_provenance.py --model gpt6astra \
    --dataset ../rexinthewild.csv --outdir results/
```

Model keys: `gpt6astra`, `gpt6astra-xhigh`, `gpt`, `gpt-xhigh`, `claude-opus5`,
`claude-opus5-max`, `gemini37flash`, `gemini`, `qwen`, `qwen32`, `lingshu`,
`medgemma`. The `-xhigh` and `-max` variants are the raised-inference-compute
arms reported in the appendix.

Images must be fetched first — see `../fetch_images.py`. Point the run at the
directory you fetched into; paths in the dataset are relative
(`images/{PMCID}_{basename}`).

## Provenance

`run_with_provenance.py` writes, alongside each result file:

- `<stem>.csv` — the model's answers
- `<stem>.meta.json` — resolved model ID, every decoding parameter, the served
  model snapshot where the provider exposes it, API version, prompts, library
  versions, timings, and SHA-256 checksums of both the dataset and the scripts
- `<stem>.log` — full stdout/stderr

> **Before sharing a `.meta.json`, check what is in it.** The sidecar records
> the host of your `AZURE_OPENAI_ENDPOINT` so that a run can be traced to the
> deployment that produced it. On a private or institutional deployment that
> hostname may identify you or your organisation. Redact it before attaching a
> sidecar to a paper, issue or public repository.

Closed-model results are not bitwise reproducible, because providers update
served checkpoints without notice. The snapshot fields are what let you tell
whether a divergence is yours or theirs. If you are comparing against the
paper's numbers, check `served_model_snapshot` first.

The interpreter used for subprocess calls can be overridden with
`REXIN_PYTHON`; it defaults to the interpreter running the wrapper.

## Scoring

```bash
python score_results.py results/gpt6astra.csv results/gemini37flash.csv
```

Blank or unparseable predictions are scored **incorrect** and reported
separately rather than dropped from the denominator. Intervals are Wilson score
intervals from `statsmodels`. Because questions carry three to five options, the
chance baseline is **22.5%**, not 25%.

## Fine-tuning

`train_cv_qwen8b.py` runs QLoRA cross-validation over `cv_folds.json`: 4-bit
NF4 base, LoRA rank 16 on the attention and MLP projections, vision tower frozen,
loss masked to the answer tokens. Folds are split by image, so no image appears
in both a training and a validation fold, and are stratified by topic tag.
