# Hyperparameters

Every decoding setting used to produce the results in the paper, in one place so
they can be compared side by side. Values are taken from the provenance record
each run wrote, not from documentation or from memory.

Nothing was tuned. Every model ran at its provider's documented defaults, with
the single exception of the raised-compute arms marked *(arm)*, which exist to
test whether more inference compute helps. All of these are reasoning models
whose providers restrict or discourage sampling parameters.

## All models side by side

`—` means the parameter does not exist for that model. **`not sent`** means the
parameter exists but was deliberately omitted, so the provider's own default
applied.

| Model | Identifier | Access | temperature | top_p / top_k | Reasoning control | max tokens |
|---|---|---|---|---|---|---|
| GPT-6 Astra | `gpt-6-astra` | Azure OpenAI | not sent | not sent | API default | — |
| GPT-6 Astra *(arm)* | `gpt-6-astra` | Azure OpenAI | not sent | not sent | `reasoning_effort: xhigh` | — |
| GPT-5.6 Sol | `gpt-5.6-sol` | Azure OpenAI | not sent | not sent | API default (medium) | — |
| GPT-5.6 Sol *(arm)* | `gpt-5.6-sol` | Azure OpenAI | not sent | not sent | `reasoning_effort: xhigh` | — |
| Claude Opus 5 | `claude-opus-5` | Anthropic API | not sent | not sent | default effort (high) | `8192` |
| Claude Opus 5 *(arm)* | `claude-opus-5` | Anthropic API | not sent | not sent | `effort: max` | `8192` |
| Gemini 3.7 Flash | `gemini-3.7-flash` | Vertex AI | `1.0` | not sent | `thinking_level: high` | — |
| Gemini 3.1 Pro | `gemini-3.1-pro-preview` | Vertex AI | `1.0` | not sent | `thinking_level: high` | — |
| Qwen3-VL-32B | `Qwen/Qwen3-VL-32B-Instruct` | local | greedy | — | — | `512` new |
| Qwen3-VL-8B | `Qwen/Qwen3-VL-8B-Instruct` | local | greedy | — | — | `512` new |
| Lingshu-7B | `lingshu-medical-mllm/Lingshu-7B` | local | greedy | — | — | `512` new |
| MedGemma-4B | `google/medgemma-4b-it` | local | greedy | — | — | `512` new |

Open-weight models all ran with `do_sample=false`, `num_beams=1`,
`max_new_tokens=512`, in bfloat16 on one NVIDIA A100 80 GB.

## Why each parameter was set that way

**Azure OpenAI (GPT-6 Astra, GPT-5.6 Sol).** `temperature` and `top_p` are not
sent at all; these deployments reject or ignore them. Reasoning effort was left
at the API default except in the arms. Both were called with API version
`2024-12-01-preview`.

**Anthropic (Claude Opus 5).** `temperature`, `top_p` and `top_k` are not sent,
as Claude Opus 4.7 and later reject them. `max_tokens` is set far above what an
answer needs because it bounds thinking and visible output *together*; a run
that truncates mid-thinking returns no answer at all. Runs that hit that limit
were repeated at a higher bound, and the override is recorded in the run's
provenance.

**Vertex AI (Gemini).** `temperature 1.0` and `thinking_level high` are both
documented defaults, and `high` is the maximum. `top_p` and `top_k` were left
unset because Google publishes no defaults for them and advises against tuning.

**Open-weight.** Greedy decoding makes these runs deterministic, and avoids the
degraded behaviour MedGemma shows under its base model's sampling defaults.

## Served snapshots

Closed models are not bitwise reproducible: providers update the checkpoint
behind a model name without notice. These are the snapshots that answered, where
the provider exposes them. Check these first if your numbers differ from the
paper's.

| Model | Snapshot at run time |
|---|---|
| GPT-6 Astra | `gpt-6-astra-2026-09-03` |
| GPT-5.6 Sol | `gpt-5.6-sol-2026-07-09` |
| Gemini 3.7 Flash | `gemini-3.7-flash` |
| Claude Opus 5 | `claude-opus-5` |

## Runtime

Two environments, because API and local models ran on different machines.

| | API runs | Open-weight runs |
|---|---|---|
| `openai` | 2.15.0 | 3.10.0 |
| `anthropic` | 0.76.0 | 1.4.0 |
| `google-genai` | 1.59.0 | 2.22.0 |
| `transformers` | 4.57.6 | **5.16.1** |
| `torch` | 2.9.1 | **2.9.1+cu129** |
| Hardware | — | NVIDIA A100 80 GB, CUDA 12.9 |

Only the open-weight column affects results; the API column is just the machine
that made the HTTP calls.

## Question generation

The generation and editing pipeline is in `prompts.md`. Its settings:

| Setting | Value |
|---|---|
| Provider | Azure OpenAI |
| API version | `2024-12-01-preview` |
| Deployment | vision-capable; default `gpt-5` |
| Parameters sent | `model`, `messages` only |
| `temperature`, `top_p`, `max_tokens` | not sent |
| Questions kept per image | top 2, ranked by summed score |
| Score components | clarity, medical relevance, medical difficulty, visual difficulty |

## Answer-option ordering

Options were shuffled once at dataset construction with a seed derived from the
question content, then held fixed, so every model saw identical ordering:

```python
seed = int(hashlib.sha256("||".join(items).encode()).hexdigest(), 16) % (2**32)
rng  = np.random.default_rng(seed)
```

Models were told to read left and right from the patient's perspective unless a
question said otherwise, and to answer with a bracketed letter. Responses with no
recoverable answer were scored incorrect rather than dropped.
