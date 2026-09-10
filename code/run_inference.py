import os
import csv
import random
import argparse
import openai
import base64
import io
import glob
import string
import json
import sys
import google.genai as genai
from google.genai import types
from google.genai.types import GenerateContentConfig
import PIL.Image as PIL_Image
import time
# Delay heavy import until needed
from transformers import AutoProcessor, AutoModelForImageTextToText, set_seed
from PIL import Image
import torch
import re

# Claude import
import anthropic

# ---------------------------------------------------------------------------
# Decoding parameters.
#
# Every parameter sent to every model is stated here rather than left to an
# API- or library-side default, so the methods section can be written from this
# block instead of from memory. Each entry names the source of its values.
#
# The open-weight models (medgemma, lingshu, qwen) are run with GREEDY decoding
# -- do_sample=False, num_beams=1 -- i.e. each model's default deterministic
# generation path. Sampling parameters are deliberately NOT passed: MedGemma is
# unreliable when given them, and greedy makes a run reproducible bit-for-bit
# rather than only under a fixed seed. This is also what the paper already
# claimed for MedGemma, now applied uniformly to all three.
# SEED is still re-applied before each generate() call as a defensive measure;
# with do_sample=False it has no effect on the output.
# ---------------------------------------------------------------------------

SEED = 0

MODEL_PARAMS = {
    # Azure OpenAI, GPT-5.6 Sol (`gpt-5.6-sol`, alias `gpt-5.6`) -- OpenAI's
    # current frontier model; vision-capable, 1.05M context, 128K max output.
    # The model actually used is whatever AZURE_OPENAI_DEPLOYMENT points at, so
    # the deployment must be created against gpt-5.6-sol.
    #
    # No sampling parameters are sent. GPT-5.x reasoning models restrict or
    # reject temperature/top_p, and the paper's claim is "default parameters",
    # so omitting them IS the setting. `reasoning_effort` is likewise left at
    # the API default. Set a value here only to deliberately override a default.
    "gpt": {"temperature": None, "top_p": None},
    "gpt-text": {"temperature": None, "top_p": None},

    # GPT-5.6 Sol at maximum available reasoning. "max" is rejected on
    # api-version 2024-12-01-preview, so "xhigh" is the ceiling here.
    "gpt-xhigh": {"temperature": None, "top_p": None, "reasoning_effort": "xhigh"},

    # GPT-6 Astra (released 2026-09-03), served snapshot gpt-6-astra-2026-09-03.
    # Same convention as the other reasoning models: no sampling parameters and
    # reasoning_effort left at the API default, so the run reflects defaults.
    # Requires AZURE_OPENAI_DEPLOYMENT=gpt-6-astra.
    "gpt6astra": {"temperature": None, "top_p": None},

    # GPT-6 Astra at maximum available reasoning. "max" is rejected on
    # api-version 2024-12-01-preview, so "xhigh" is the ceiling. The main
    # gpt6astra arm runs at the API default, so this is the "pay more" arm.
    "gpt6astra-xhigh": {"temperature": None, "top_p": None, "reasoning_effort": "xhigh"},

    # Anthropic Claude Opus 5 (`claude-opus-5`).
    #
    # temperature / top_p / top_k were REMOVED on Opus 4.7 and later -- sending
    # any of them returns a 400. Behaviour is steered by prompting only, so the
    # old temperature=0 determinism setting no longer exists as an option.
    #
    # Thinking is ON by default on Opus 5 (omitting `thinking` runs adaptive),
    # and `effort` defaults to "high". Both are left at their defaults.
    #
    # max_tokens caps thinking + visible text together, so the old 512 would now
    # risk truncating the answer while the model is still thinking. Raised to
    # give the default adaptive thinking room; the visible answer is ~5 tokens.
    "claude-opus5": {"model_id": "claude-opus-5", "max_tokens": 8192},

    # Opus 5 at maximum thinking. Effort is set through output_config.effort
    # (thinking.effort and a top-level effort are both rejected with a 400).
    "claude-opus5-max": {"model_id": "claude-opus-5", "max_tokens": 8192,
                         "effort": "max"},

    # Google Gemini 3.1 Pro (`gemini-3.1-pro-preview`) -- Google's strongest
    # reasoning model. Still preview-tagged, as gemini-3-pro-preview was.
    #
    # Google documents temperature 1.0 as the default and "strongly recommends"
    # leaving it there -- lowering it can cause looping or degraded reasoning.
    # thinking_level defaults to "high". Both are passed at their documented
    # defaults so the request is self-documenting without changing behaviour.
    # top_p / top_k are deliberately NOT set: Gemini 3.x publishes no defaults
    # for them and Google advises against tuning them.
    "gemini": {"model_id": "gemini-3.1-pro-preview", "temperature": 1.0,
               "thinking_level": "high"},

    # --- Appendix arms: "what if you had paid more, or picked a newer model?" ---
    # Gemini 3.7 Flash (2026-08-13). Newer than 3.1 Pro but a cheaper tier; the
    # Pro line has not moved past 3.1, so this is Google's newest model overall.
    "gemini37flash": {"model_id": "gemini-3.7-flash", "temperature": 1.0,
                      "thinking_level": "high"},

    # google/medgemma-4b-it -- greedy, matching the model card's own usage
    # example. Passing Gemma 3's sampling defaults here makes MedGemma unreliable.
    "medgemma": {
        "model_id": "google/medgemma-4b-it",
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 512,
    },

    # lingshu-medical-mllm/Lingshu-7B -- greedy, for parity with the other
    # open-weight models and for bit-for-bit reproducibility. (The model card
    # suggests temperature 0.7 / top_p 1.0 for open-ended generation; this task
    # wants a single bracketed letter, so greedy is the appropriate default.)
    # Built on Qwen2.5-VL, so it loads through Qwen2_5_VLForConditionalGeneration.
    "lingshu": {
        "model_id": "lingshu-medical-mllm/Lingshu-7B",
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 512,
    },

    # Qwen/Qwen3-VL-8B-Instruct -- greedy, same reasoning as Lingshu.
    "qwen": {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 512,
    },

    # --- 32B tier: scale-matched pair (33.36B vs 33.45B), needs ~67GB in bf16,
    # so an A100 80GB. Same greedy settings as the 8B tier so the two scales are
    # directly comparable.
    "qwen32": {
        "model_id": "Qwen/Qwen3-VL-32B-Instruct",
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 512,
    },
    "lingshu32": {
        "model_id": "lingshu-medical-mllm/Lingshu-32B",
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 512,
    },
}

# Single source of truth for --model, so the CLI choices, the dispatch table and
# the error text cannot drift apart (which is how 'llava-med' ended up
# implemented but unreachable from the command line).
SUPPORTED_MODELS = list(MODEL_PARAMS)

# Local models are large; load each one once and reuse it across questions.
_local_model_cache = {}
_consecutive_local_failures = 0


def azure_sampling_kwargs(model_key):
    """Sampling parameters for an Azure OpenAI call, dropping any set to None."""
    return {k: v for k, v in MODEL_PARAMS[model_key].items() if v is not None}

SYSTEM_PROMPT = """You are an expert clinician."""
USER_PROMPT = (
    "Given the following medical image and a multiple-choice question, select the single best answer from the choices.\n"
    "ONLY output your FINAL answer, which must be placed inside curly brackets, e.g. {A}. Provide NO extra words, reasoning or explanation outside the curly brackets.\n"
    "When asked about left/right, answer from the patient's viewpoint unless otherwise stated.\n"
)

# ---- Azure OpenAI API setup ----
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_api_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT") or "gpt-5.6-sol"
# 2024-12-01-preview predates GPT-5.6 and may not expose it. Override with
# AZURE_OPENAI_API_VERSION to whatever version the deployment requires.
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION") or "2024-12-01-preview"

# Do not print the key itself -- these logs get redirected to files and pasted
# into issues.
print(f"Azure API key: {'set' if azure_api_key else 'NOT SET'}", flush=True)
print(f"Azure API endpoint: {azure_api_endpoint}", flush=True)
print(f"Azure deployment name: {azure_deployment_name}", flush=True)
print(f"Azure API version: {azure_api_version}", flush=True)

try:
    if azure_api_key and azure_api_endpoint:
        azure_openai_client = openai.AzureOpenAI(
            api_key=azure_api_key,
            azure_endpoint=azure_api_endpoint,
            api_version=azure_api_version,
        )
except Exception as e:
    print(f"Warning: Could not initialize Azure OpenAI client: {e}", flush=True)
    exit(1)

def load_llava_med():
    global _llava_cache
    if _llava_cache is None:
        disable_torch_init()
        model_name = get_model_name_from_path(LLAVA_MED_MODEL_ID)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            LLAVA_MED_MODEL_ID,
            model_base=None,
            model_name=model_name,
            device_map="auto",
        )
        _llava_cache = tokenizer, model, image_processor, context_len
    return _llava_cache

def image_to_data_url(image_path):
    ext = os.path.splitext(str(image_path))[1].lower()
    mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def find_image_path(image_file_name, pmcid, image_dir):
    """
    Find a local image whose name contains both pmcid and image_file_name.
    """
    pmcid_str = str(pmcid)
    candidates = glob.glob(os.path.join(image_dir, "*"))
    for path in candidates:
        fname = os.path.basename(path)
        if image_file_name.lower() in fname.lower() and pmcid_str.lower() in fname.lower():
            return path
    return None

import sys

def extract_bracketed_answer(text, valid_letters="ABCDE"):
    """
    Pull the chosen option letter out of a model response.

    The prompt asks for {A}, but models comply inconsistently: Lingshu-7B emitted
    a bare letter or "D." 53% of the time and [D] 23% of the time, using {D} for
    only 24%. A curly-brace-only parser silently discarded 76% of its answers --
    and because unparsed rows are dropped rather than scored wrong, the survivors
    were a biased subset. This parser accepts the conventions models actually
    use, so that step is automatic and reproducible rather than manual.

    Patterns are tried in order of decreasing explicitness. Within whichever
    pattern first matches, ALL matches are collected: if they disagree -- as in
    "the answer is [a] or [b]" -- the response names more than one answer, and
    "" is returned rather than taking whichever appeared first. Picking the
    first would score a hedged response as a confident one, and would depend on
    word order rather than on content.

    Returns "" when nothing unambiguous is present, so unreadable or
    self-contradicting output stays visible as a failure instead of being
    guessed at. Callers score "" as incorrect; it is never dropped from the
    denominator.
    """
    if not text:
        return ""
    t = text.strip()
    L = f"[{valid_letters}]"

    patterns = [
        rf'\{{\s*({L})\s*[\}}\.\)]',      # {A}   and the {A. / {A) typos
        rf'\[\s*({L})\s*\]',                # [A]
        rf'\(\s*({L})\s*\)',                # (A)
        rf'(?:answer|choice|option)\s*(?:is|:)?\s*[\{{\[\(]?\s*({L})\b',  # Answer: A
        rf'^\s*({L})\s*\.?\s*$',            # the WHOLE reply is "A" or "A."
    ]
    for pat in patterns:
        found = re.findall(pat, t, flags=re.IGNORECASE | re.MULTILINE)
        if not found:
            continue
        letters = {m.upper() for m in found}
        if len(letters) > 1:
            return ""          # names more than one answer -> no answer
        return letters.pop()

    # Deliberately no further guessing. A letter heading longer prose
    # ("A. Cleft lip") is NOT auto-assigned -- it goes to manual verification.
    # Returning "" keeps it visible instead of inventing a choice.
    return ""

# This utility ensures choices are injected as a formatted string, not a Python list or extra-quoted.
def format_choices_raw(choices_raw):
    """
    Takes the choices_raw field from CSV (string, already with newlines/labels),
    and formats it for prompt display as a single block of text,
    stripping unwanted surrounding brackets or quotes/JSON artifacts.
    """
    # Remove surrounding [] if choices_raw is a single string in brackets
    if isinstance(choices_raw, str) and choices_raw.startswith("[") and choices_raw.endswith("]"):
        without_brackets = choices_raw[1:-1]
        # Remove surrounding single/double quotes if present
        if (without_brackets.startswith("'") and without_brackets.endswith("'")) or (without_brackets.startswith('"') and without_brackets.endswith('"')):
            without_brackets = without_brackets[1:-1]
        # Replace escaped newlines if present
        without_brackets = without_brackets.replace("\\n", "\n")
        return without_brackets.strip()
    elif isinstance(choices_raw, str):
        return choices_raw.strip()
    else:
        return str(choices_raw).strip()

def generate_gpt_answer(question, choices, image_path, model_key="gpt"):
    gpt_model_key = model_key
    data_url = image_to_data_url(image_path)
    full_user_prompt = (
        f"{USER_PROMPT}\n"
        f"Question: {question}\n"
        f"Choices:\n{choices}"
    )
    system_message = {
        "role": "system",
        "content": [{"type": "text", "text": SYSTEM_PROMPT}]
    }
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": full_user_prompt
            },
            {"type": "image_url", "image_url": {"url": data_url}}
        ]
    }
    
    try:
        response = azure_openai_client.chat.completions.create(
            model=azure_deployment_name,
            messages=[system_message, user_message],
            **azure_sampling_kwargs(gpt_model_key)
        )
        content = response.choices[0].message.content.strip()
        json_str = None
        # Try to find a JSON object in the response if there is external reasoning
        matches = re.findall(r'({\s*"answer"\s*:\s*".*?"\s*})', content, re.DOTALL)
        if matches:
            json_str = matches[0]
        else:
            # fallback: maybe full content is JSON
            json_str = content

        try:
            answer_json = json.loads(json_str)
            full_answer = str(answer_json.get("answer", "")).strip()
        except Exception:
            # fallback: just return the raw content if not JSON
            full_answer = content
        model_answer = extract_bracketed_answer(full_answer)
        return model_answer, full_user_prompt, full_answer
    except Exception as e:
        print(f"[error][answer-generation] {e} for {image_path}", flush=True)
        time.sleep(10)
        return "", full_user_prompt, ""

def generate_gpt_text_answer(question, choices, image_path):
    gpt_model_key = "gpt-text"
    # This purposely ignores the image and instructs the model not to use it.
    full_user_prompt = (
        f"{USER_PROMPT}\n"
        f"Question: {question}\n"
        f"Choices:\n{choices}"
    )
    system_message = {
        "role": "system",
        "content": [
            {"type": "text", "text": """Here is a medical visual-question answering task.\n
            Do your best to answer the question without seeing the image, by instead using your medical knowledge and test-taking skills.\n
            Given the following multiple-choice question, select the single best answer from the choices.\n
            ONLY output your FINAL answer, which must be placed inside curly brackets, e.g. {A}. Provide NO extra words, reasoning or explanation outside the curly brackets.\n
            When asked about left/right, answer from the patient's viewpoint unless otherwise stated.\n"""
            }
        ]
    }
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": full_user_prompt
            },
        ]
    }
    
    try:
        response = azure_openai_client.chat.completions.create(
            model=azure_deployment_name,
            messages=[system_message, user_message],
            **azure_sampling_kwargs(gpt_model_key)
        )
        content = response.choices[0].message.content.strip()
        json_str = None
        # Try to find a JSON object in the response if there is external reasoning
        matches = re.findall(r'({\s*"answer"\s*:\s*".*?"\s*})', content, re.DOTALL)
        if matches:
            json_str = matches[0]
        else:
            # fallback: maybe full content is JSON
            json_str = content

        try:
            answer_json = json.loads(json_str)
            full_answer = str(answer_json.get("answer", "")).strip()
        except Exception:
            # fallback: just return the raw content if not JSON
            full_answer = content
        model_answer = extract_bracketed_answer(full_answer)
        return model_answer, full_user_prompt, full_answer
    except Exception as e:
        print(f"[error][answer-generation] {e} for {image_path}", flush=True)
        return "", full_user_prompt, ""

def load_local_vlm(model_key):
    """
    Load (and cache) one of the locally-run open-weight vision-language models.

    Each model needs a different model class, so the class is imported lazily --
    Qwen3-VL in particular requires a recent transformers build, and importing it
    unconditionally would break the API-only backends on older installs.
    """
    if model_key in _local_model_cache:
        return _local_model_cache[model_key]

    model_id = MODEL_PARAMS[model_key]["model_id"]

    if model_key == "medgemma":
        model_cls = AutoModelForImageTextToText
    elif model_key in ("lingshu", "lingshu32"):
        # Both Lingshu sizes are Qwen2.5-VL derivatives (confirmed from config.json:
        # architectures = Qwen2_5_VLForConditionalGeneration).
        from transformers import Qwen2_5_VLForConditionalGeneration
        model_cls = Qwen2_5_VLForConditionalGeneration
    elif model_key in ("qwen", "qwen32"):
        # Requires transformers >= 4.57 (or a source build) for Qwen3-VL.
        from transformers import Qwen3VLForConditionalGeneration
        model_cls = Qwen3VLForConditionalGeneration
    else:
        raise ValueError(f"No local loader defined for model '{model_key}'")

    print(f"[load][{model_key}] loading {model_id} (first question only)", flush=True)
    model = model_cls.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    _local_model_cache[model_key] = (model, processor)
    return model, processor


def generate_local_vlm_answer(model_key, question, choices, image_path):
    """
    Shared inference path for the open-weight models (medgemma, lingshu, qwen).

    All three take the same chat-template messages and differ only in the
    decoding parameters recorded in MODEL_PARAMS. The seed is re-applied before
    every generate() call so results do not depend on how many questions were
    answered earlier in the run.
    """
    global _consecutive_local_failures
    params = MODEL_PARAMS[model_key]

    try:
        model, processor = load_local_vlm(model_key)
    except Exception as e:
        # Fail fast. A load failure is never transient (gated repo, missing
        # token, OOM), and retrying per question silently produced 960 empty
        # answers instead of one clear error.
        print(f"[error][{model_key}-load] {e}", flush=True)
        print(f"[fatal] cannot load {model_key}; aborting run.", flush=True)
        sys.exit(2)

    try:
        # Convert to RGB: some PubMed figures are palette-mode or CMYK, which
        # the image processors will not accept as-is.
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[error][{model_key}-image] {e}", flush=True)
        return "", f"Failed to open image: {e}", ""

    full_user_prompt = (
        f"{USER_PROMPT}\n"
        f"Question: {question}\n"
        f"Choices:\n{choices}"
    )

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": full_user_prompt},
                {"type": "image", "image": image}
            ]
        }
    ]

    # Only the decoding keys go to generate(); model_id is bookkeeping.
    gen_kwargs = {k: v for k, v in params.items() if k != "model_id"}

    try:
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]
        set_seed(SEED)
        with torch.inference_mode():
            generation = model.generate(**inputs, **gen_kwargs)
            generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True).strip()
        # Save full answer
        full_answer = decoded
        model_answer = extract_bracketed_answer(full_answer)
        _consecutive_local_failures = 0
        return model_answer, full_user_prompt, full_answer
    except Exception as e:
        # A systematic fault (bad dependency, OOM, wrong dtype) fails on EVERY
        # question. Without this guard a run "completes" with 960 empty answers
        # and a header-only CSV, which looks like a finished job. Abort early.
        _consecutive_local_failures += 1
        print(f"[error][{model_key}-generate] {e}", flush=True)
        if _consecutive_local_failures >= 5:
            print(f"[fatal] {model_key}: {_consecutive_local_failures} consecutive "
                  f"generation failures -- aborting rather than emitting empty rows.",
                  flush=True)
            sys.exit(3)
        return "", full_user_prompt, ""


def generate_medgemma_answer(question, choices, image_path):
    return generate_local_vlm_answer("medgemma", question, choices, image_path)


def generate_lingshu_answer(question, choices, image_path):
    return generate_local_vlm_answer("lingshu", question, choices, image_path)


def generate_qwen_answer(question, choices, image_path):
    return generate_local_vlm_answer("qwen", question, choices, image_path)

_gemini_client = None
_claude_client = None


def get_gemini_client():
    """
    Build (once) a genai client for whichever auth path is configured.

    Two supported paths:
      * Vertex AI  -- GOOGLE_GENAI_USE_VERTEXAI=true plus GOOGLE_CLOUD_PROJECT /
        GOOGLE_CLOUD_LOCATION. Credentials come from Application Default
        Credentials (`gcloud auth application-default login`), NOT an API key.
      * Gemini API -- GOOGLE_API_KEY.

    The client used to be rebuilt on every question; it is now cached.
    """
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client

    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() in ("1", "true", "yes")
    api_key = os.environ.get("GOOGLE_API_KEY")

    # Without an explicit timeout a stalled request blocks forever: a 960-question
    # run hung for 17h on question 901 with the socket open and no CPU activity.
    # A per-request timeout turns that into an ordinary error, and because failed
    # rows are never written they are simply retried on the next run.
    timeout_ms = int(os.environ.get("GEMINI_TIMEOUT_MS", "180000"))
    http_opts = types.HttpOptions(timeout=timeout_ms)

    try:
        if use_vertex:
            project = os.environ.get("GOOGLE_CLOUD_PROJECT")
            location = os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"
            if not project:
                print("[error][gemini] GOOGLE_GENAI_USE_VERTEXAI is set but "
                      "GOOGLE_CLOUD_PROJECT is missing.", flush=True)
                sys.exit(1)
            print(f"[gemini] Vertex AI project={project} location={location} (ADC) "
                  f"timeout={timeout_ms}ms", flush=True)
            _gemini_client = genai.Client(vertexai=True, project=project, location=location,
                                          http_options=http_opts)
        elif api_key:
            print(f"[gemini] Gemini API via GOOGLE_API_KEY timeout={timeout_ms}ms", flush=True)
            _gemini_client = genai.Client(api_key=api_key, http_options=http_opts)
        else:
            print("[error][gemini] No credentials. Set GOOGLE_GENAI_USE_VERTEXAI=true "
                  "with GOOGLE_CLOUD_PROJECT (ADC), or set GOOGLE_API_KEY.", flush=True)
            sys.exit(1)
    except Exception as e:
        print(f"[error][gemini-client] Failed to initialize Gemini client: {e}", flush=True)
        sys.exit(1)
    return _gemini_client


def generate_gemini_answer(question, choices, image_path, model_key="gemini"):
    client = get_gemini_client()

    try:
        image = PIL_Image.open(image_path)
    except Exception as e:
        print(f"[error][gemini-image] {e}", flush=True)
        return "", f"Failed to open image: {e}", ""
    
    # Per-call pause to stay inside rate limits. 5s x 960 questions is ~80 min of
    # pure sleeping; override with GEMINI_SLEEP_SECONDS once quota is known good.
    time.sleep(float(os.environ.get("GEMINI_SLEEP_SECONDS", "5")))
    full_user_prompt = (
        f"{USER_PROMPT}\n"
        f"Question: {question}\n"
        f"Choices:\n{choices}"
    )

    contents = [
        full_user_prompt,
        image
    ]
    try:
        params = MODEL_PARAMS[model_key]
        response = client.models.generate_content(
            model=params["model_id"],
            contents=contents,
            config=GenerateContentConfig(
                system_instruction=[
                    SYSTEM_PROMPT,
                ],
                temperature=params["temperature"],
                # thinking_level defaults to "high" on Gemini 3; set explicitly
                # so the run does not depend on an undocumented server default.
                # Do not also pass thinking_budget -- Gemini 3 errors on both.
                thinking_config=types.ThinkingConfig(
                    thinking_level=params["thinking_level"]
                ),
            ),
        )
        content = str(getattr(response, "text", "")).strip()
        # Save full answer
        full_answer = content
        model_answer = extract_bracketed_answer(full_answer)

        if model_answer.lower() == "none":
            print("[warning][gemini] Answer is 'None'. Sleeping for 2 minutes before returning empty answer.", flush=True)
            time.sleep(120)
            return "", full_user_prompt, full_answer

        return model_answer, full_user_prompt, full_answer
    except Exception as e:
        print(f"[error][gemini-generate] {e}", flush=True)
        return "", full_user_prompt, ""

def generate_claude_opus45_answer(question, choices, image_path, model_key="claude-opus5"):
    # Uses anthropic/claude-agent-sdk (Python)
    claude_api_key = os.environ.get("CLAUDE_API_KEY", None)
    if not claude_api_key:
        print("[error][claude-opus5] CLAUDE_API_KEY environment variable is not set!", flush=True)
        sys.exit(1)
    global _claude_client
    if _claude_client is None:
        # Identity-linked keys (provisioned against a user rather than a single
        # workspace) are rejected with a 400 unless every request names the
        # workspace it bills to. Discover it with:
        #   GET https://api.anthropic.com/v1/organizations/workspaces
        headers = {}
        ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
        if ws:
            headers["anthropic-workspace-id"] = ws
            print(f"[claude] workspace {ws}", flush=True)
        try:
            _claude_client = anthropic.Anthropic(api_key=claude_api_key,
                                                 default_headers=headers or None)
        except Exception as e:
            print(f"[error][claude-init] Failed to initialize Claude client: {e}", flush=True)
            sys.exit(1)
    client = _claude_client

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    except Exception as e:
        print(f"[error][claude-image] {e}", flush=True)
        return "", f"Failed to open image: {e}", ""

    full_user_prompt = (
        f"{USER_PROMPT}\n"
        f"Question: {question}\n"
        f"Choices:\n{choices}"
    )

    # Anthropic Claude Opus 4.5 supports vision
    try:
        # See https://docs.anthropic.com/claude/reference/messages-post
        # Fix: Place system prompt at top-level, not as a message role.
        params = MODEL_PARAMS[model_key]
        _extra = ({"output_config": {"effort": params["effort"]}}
                  if params.get("effort") else {})
        message = client.messages.create(
            model=params["model_id"],
            max_tokens=params["max_tokens"],
            extra_body=_extra,
            # No temperature/top_p/top_k: removed on Opus 4.7+ and rejected with
            # a 400. Thinking and effort are left at their Opus 5 defaults
            # (adaptive thinking on, effort "high").
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_user_prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png" if image_path.lower().endswith(".png") else "image/jpeg",
                                "data": base64.b64encode(image_bytes).decode("utf-8")
                            }
                        }
                    ]
                }
            ]
        )
        # Opus 5 runs safety classifiers that can decline a request: this comes
        # back as a normal HTTP 200 with stop_reason "refusal" and empty or
        # partial content. Record it explicitly so it is never mistaken for a
        # wrong answer or a network blip.
        #
        # Server-side `fallbacks` is deliberately NOT enabled here: silently
        # re-serving a refused benchmark question on a different model would
        # attribute another model's answer to Claude Opus 5.
        if getattr(message, "stop_reason", None) == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None)
            print(f"[refusal][claude] stop_reason=refusal category={category}", flush=True)
            return "", full_user_prompt, f"[REFUSAL] category={category}", 0

        # With thinking on by default, max_tokens covers thinking + visible text.
        # A truncated turn would otherwise silently look like an empty answer.
        if getattr(message, "stop_reason", None) == "max_tokens":
            print(
                f"[warning][claude] hit max_tokens ({params['max_tokens']}) for {image_path}; "
                "answer may be truncated -- consider raising max_tokens",
                flush=True,
            )

        # Get response content (Claude returns list of text blocks).
        # Thinking blocks are skipped by the type check: on Opus 5 the raw chain
        # of thought is never returned and `display` defaults to "omitted".
        content = ""
        parts = []
        if hasattr(message, "content"):
            for p in message.content:
                if getattr(p, "type", None) == "text":
                    parts.append(p.text)
        content = "\n".join(parts).strip() if parts else ""
        full_answer = content
        model_answer = extract_bracketed_answer(full_answer)

        if model_answer.lower() == "none":
            print("[warning][claude] Answer is 'None'. Returning empty answer.", flush=True)
            return "", full_user_prompt, full_answer, 0

        # Get Claude tokens used (input + output) if present
        num_tokens = 0
        try:
            if hasattr(message, "usage") and message.usage is not None:
                # Claude's .usage is {"input_tokens": int, "output_tokens": int}
                in_toks = getattr(message.usage, "input_tokens", 0)
                out_toks = getattr(message.usage, "output_tokens", 0)
                if in_toks is not None and out_toks is not None:
                    num_tokens = int(in_toks) + int(out_toks)
        except Exception:
            num_tokens = 0
        return model_answer, full_user_prompt, full_answer, num_tokens
    except Exception as e:
        print(f"[error][claude-generate] {e}", flush=True)
        return "", full_user_prompt, "", 0

def generate_llava_med_answer(question, choices, image_path):
    tokenizer, model, image_processor, _ = load_llava_med()

    full_user_prompt = (
        f"{USER_PROMPT}\n"
        f"Question: {question}\n"
        f"Choices:\n{choices}"
    )

    qs = DEFAULT_IMAGE_TOKEN + "\n" + full_user_prompt
    if getattr(model.config, "mm_use_im_start_end", False):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + full_user_prompt

    conv = conv_templates["mistral_instruct"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)[0]

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            do_sample=False,
            temperature=0.0,
            max_new_tokens=100,
            use_cache=True,
        )

    full_answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    model_answer = extract_bracketed_answer(full_answer)
    return model_answer, full_user_prompt, full_answer

def normalize_ans(ans):
    if ans is None:
        return ""
    ans = ans.replace("•", "") # remove any bullet points
    ans = ans.strip()
    return "".join(ans.lower().split())
def strip_punct(text):
    return text.translate(str.maketrans('', '', string.punctuation)).strip()
def calculate_accuracy(csv_path, answer_column, question_column="edited_question", answer_gold_column="edited_answer"):
    correct = 0
    total = 0
    incorrect_cases = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            gold = row.get(answer_gold_column, '') 
            pred = row.get(answer_column, '')
            if not gold or not pred:
                continue
            gold_norm = normalize_ans(gold)
            pred_norm = normalize_ans(pred)
            if strip_punct(gold_norm) == strip_punct(pred_norm):
                correct += 1
            else:
                incorrect_cases.append({
                    "row": idx,
                    "pmcid": row.get("pmcid",""),
                    "image_file_name": row.get("image_file_name",""),
                    "question": row.get(question_column,""),
                    "gold": gold,
                    "pred": pred
                })
            total += 1
    accuracy = correct / total if total > 0 else 0.0
    print(f"Accuracy: {accuracy:.4f} ({correct} / {total})")
    if incorrect_cases:
        print(f"Sample incorrect cases (showing up to 5):")
        for case in incorrect_cases[:5]:
            print(f"Row {case['row']}: gold='{case['gold']}' pred='{case['pred']}' (pmcid={case['pmcid']}, img={case['image_file_name']})")
    return accuracy

def main(
    input_csv_path,
    output_csv_path,
    answer_column,
    image_dir,
    model,
    prompt_column,
    calc_acc_only=False,
    question_column="final_question",
    choices_column="final_choices",
    answer_gold_column="final_answer",
):
    # Accuracy-only mode
    if calc_acc_only:
        calculate_accuracy(
            output_csv_path or input_csv_path,
            question_column=question_column,
            answer_column=answer_column,
            answer_gold_column=answer_gold_column,
        )
        return

    output_csv_path = output_csv_path or input_csv_path

    # --- State for existing output file / resume logic ---
    existing_triples_with_answer = set()
    output_has_answer_field = False
    exist_fieldnames = None  # header of existing output CSV, if any

    # Build the unformatted answer column name
    unformatted_answer_col = f"{answer_column}_unformatted"

    if os.path.exists(output_csv_path) and os.path.getsize(output_csv_path) > 0:
        with open(output_csv_path, newline="", encoding="utf-8") as f_exist:
            reader_exist = csv.DictReader(f_exist)
            exist_fieldnames = reader_exist.fieldnames if reader_exist.fieldnames else []
            output_has_answer_field = (
                answer_column in exist_fieldnames if exist_fieldnames else False
            )
            for row in reader_exist:
                pmcid = row.get("pmcid")
                img_name = row.get("image_file_name")
                question_val = row.get(question_column, "")
                ans_val = row.get(answer_column, "") if output_has_answer_field else ""
                if (
                    pmcid
                    and img_name
                    and question_val
                    and ans_val
                    and ans_val.strip() != ""
                ):
                    existing_triples_with_answer.add((pmcid, img_name, question_val))

    correct_so_far = 0
    total_so_far = 0

    def print_interim_accuracy():
        accuracy = correct_so_far / total_so_far if total_so_far > 0 else 0.0
        print(
            f"[progress] Interim accuracy: {accuracy:.4f} ({correct_so_far} / {total_so_far})",
            flush=True,
        )

    # For Claude: add a tokens column if not already present!
    def maybe_add_claude_tokens_column(fieldnames, model, prompt_column):
        claude_tokens_col = "claude_tokens_used"
        if model in ("claude-opus5", "claude-opus5-max"):
            if claude_tokens_col not in fieldnames:
                # Place tokens column just after prompt_column if possible
                try:
                    idx = fieldnames.index(prompt_column)
                    return fieldnames[:idx+1] + [claude_tokens_col] + fieldnames[idx+1:]
                except Exception:
                    # On any failure, just append
                    return fieldnames + [claude_tokens_col]
        return fieldnames

    # Ensure unformatted column is added
    def maybe_add_unformatted_column(fieldnames, colname):
        if colname not in fieldnames:
            # Place just after answer_column if present, else append
            try:
                idx = fieldnames.index(answer_column)
                return fieldnames[:idx+1] + [colname] + fieldnames[idx+1:]
            except Exception:
                return fieldnames + [colname]
        return fieldnames

    with open(input_csv_path, newline="", encoding="utf-8") as fin, open(
        output_csv_path, "a", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        input_fieldnames = reader.fieldnames or []

        # Decide header schema for writer:
        # - If output already has a header, reuse it (append-safe).
        # - Otherwise, derive from input CSV (passthrough + answer/prompt(+tokens)).
        if exist_fieldnames:
            output_fieldnames = exist_fieldnames
        else:
            passthrough_cols = [
                col for col in input_fieldnames if col not in {answer_column, prompt_column, unformatted_answer_col}
            ]
            output_fieldnames = passthrough_cols + [answer_column, unformatted_answer_col, prompt_column]
        # Only insert tokens col for Claude
        output_fieldnames = maybe_add_claude_tokens_column(output_fieldnames, model, prompt_column)
        # Add unformatted col, unless already present
        output_fieldnames = maybe_add_unformatted_column(output_fieldnames, unformatted_answer_col)

        writer = csv.DictWriter(fout, fieldnames=output_fieldnames)

        need_header = os.path.getsize(output_csv_path) == 0
        if need_header:
            writer.writeheader()

        for idx, row in enumerate(reader):
            pmcid = row.get("pmcid", "").strip()
            img_name = row.get("image_file_name", "").strip()
            question = row.get(question_column, "").strip()
            choices_raw = row.get(choices_column, "").strip()

            # Skip if we already have an answer for this triple in the output
            if (pmcid, img_name, question) in existing_triples_with_answer:
                print(
                    f"[skip][already found answer_column] For pmcid={pmcid}, image={img_name}, question={question!r} -> skipping, already found",
                    flush=True,
                )
                continue

            # Basic structural checks (question, choices, image)
            if not question or not choices_raw:
                print(
                    f"[skip][missing Q or choices] idx={idx} {pmcid} {img_name}",
                    flush=True,
                )
                continue

            # Format choices string for prompt display per instructions
            choices_str = format_choices_raw(choices_raw)

            if not choices_str:
                print(
                    f"[skip][no formatted choices] idx={idx} {pmcid} {img_name}",
                    flush=True,
                )
                continue

            img_path = find_image_path(img_name, pmcid, image_dir)
            if img_path is None:
                print(
                    f"[skip][image not found] {img_name} for pmcid={pmcid}",
                    flush=True,
                )
                continue

            # Get model answer; provide full/unformatted answer as well
            claude_tokens_used = ""
            answer_val = ""
            unformatted_val = ""
            if model in ("gpt", "gpt-xhigh", "gpt6astra", "gpt6astra-xhigh"):
                answer_val, prompt_used, unformatted_val = generate_gpt_answer(
                    question, choices_str, img_path, model_key=model
                )
            elif model == "gpt-text":
                answer_val, prompt_used, unformatted_val = generate_gpt_text_answer(
                    question, choices_str, img_path
                )
            elif model == "medgemma":
                answer_val, prompt_used, unformatted_val = generate_medgemma_answer(
                    question, choices_str, img_path
                )
            elif model == "lingshu":
                answer_val, prompt_used, unformatted_val = generate_lingshu_answer(
                    question, choices_str, img_path
                )
            elif model == "qwen":
                answer_val, prompt_used, unformatted_val = generate_qwen_answer(
                    question, choices_str, img_path
                )
            elif model in ("qwen32", "lingshu32"):
                answer_val, prompt_used, unformatted_val = generate_local_vlm_answer(
                    model, question, choices_str, img_path
                )
            elif model in ("gemini", "gemini37flash"):
                answer_val, prompt_used, unformatted_val = generate_gemini_answer(
                    question, choices_str, img_path, model_key=model
                )
            elif model in ("claude-opus5", "claude-opus5-max"):
                answer_res = generate_claude_opus45_answer(
                    question, choices_str, img_path, model_key=model
                )
                # The new return is always (main_answer, prompt, unformatted, tokens)
                if len(answer_res) == 4:
                    answer_val, prompt_used, unformatted_val, claude_tokens_used = answer_res
                else:
                    # Fallback for compatibility
                    answer_val, prompt_used = answer_res[:2]
                    unformatted_val = ""
                    claude_tokens_used = answer_res[2] if len(answer_res) > 2 else ""
            elif model == "llava-med":
                answer_val, prompt_used, unformatted_val = generate_llava_med_answer(
                    question, choices_str, img_path
                )
            else:
                print(
                    f"[error][invalid model] Model '{model}' is not supported. "
                    f"Supported models: {', '.join(SUPPORTED_MODELS)}",
                    flush=True,
                )
                sys.exit(1)

            print(f"Question: {question}", flush=True)
            print(f"{model} answer length: {len(str(answer_val))}", flush=True)
            print(f"{model} answer (first []): {answer_val}", flush=True)
            print(f"{model} unformatted answer: {unformatted_val}", flush=True)
            gold = row.get(answer_gold_column, "")
            print(f"Ground-truth answer: {gold}", flush=True)
            print(f"--------------------------------", flush=True)

            # --- Only treat empty/None/error-like answers/prompts as special ---
            prompt_used = (prompt_used or "").strip()
            answer_val = (answer_val or "").strip()
            if isinstance(claude_tokens_used, int):
                claude_tokens_used = str(claude_tokens_used)
            unformatted_val = (unformatted_val or "").strip()

            if not prompt_used:
                print(
                    f"[skip][EMPTY PROMPT] idx={idx} {pmcid} {img_name}",
                    flush=True,
                )
                continue

            if not answer_val or answer_val.lower() == "none":
                print(
                    f"[skip][EMPTY ANSWER] idx={idx} {pmcid} {img_name}",
                    flush=True,
                )
                continue

            # Live accuracy tracking (best-effort, using raw answer_val)
            if gold:
                gold_norm = normalize_ans(gold)
                pred_norm = normalize_ans(answer_val)
                total_so_far += 1
                if strip_punct(gold_norm) == strip_punct(pred_norm):
                    correct_so_far += 1
                print_interim_accuracy()

            # Build row to match *output_fieldnames* exactly
            new_row = {}
            for col in output_fieldnames:
                if col == answer_column:
                    new_row[col] = answer_val
                elif col == unformatted_answer_col:
                    new_row[col] = unformatted_val
                elif col == prompt_column:
                    new_row[col] = prompt_used
                elif col == "claude_tokens_used" and model == "claude-opus5":
                    new_row[col] = claude_tokens_used
                else:
                    # For any other column, copy from input row if present, else blank
                    new_row[col] = row.get(col, "")

            print("writing row to output csv", flush=True)
            writer.writerow(new_row)
            fout.flush()

            # Mark this triple as answered
            existing_triples_with_answer.add((pmcid, img_name, question))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate model answers for ReXInTheWild multiple-choice questions. "
            f"Backends: {', '.join(SUPPORTED_MODELS)}."
        )
    )
    parser.add_argument("input_csv", help="Path to the input CSV of MC questions")
    parser.add_argument("output_csv", help="Path to output CSV")
    parser.add_argument(
        "model",
        choices=SUPPORTED_MODELS,
        help=f"Which model backend to use (required). One of: {', '.join(SUPPORTED_MODELS)}"
    )
    parser.add_argument(
        "image_dir",
        nargs="?",
        default="question_retrieved_images",
        help="Directory containing retrieved images (default: question_retrieved_images)"
    )
    parser.add_argument(
        "--calculate_accuracy",
        action="store_true",
        help="Just calculate accuracy using the answer fields and AImodel_answer. No answer generation."
    )
    parser.add_argument(
        "--question_column",
        default="final_question",
        help="Column name to use as the question (default: edited_question)"
    )
    parser.add_argument(
        "--choices_column",
        default="final_choices",
        help="Column name to use as the choices (default: edited_choices)"
    )
    parser.add_argument(
        "--answer_column",
        default=None,
        help="Column to write AI model answers into. Required; no default."
    )
    parser.add_argument(
        "--answer_gold_column",
        default="final_answer",
        help="Column name containing the gold/ground truth answer (default: edited_answer)"
    )
    parser.add_argument(
        "--prompt_column",
        default=None,
        help="Column to record the prompt sent to the model (default: {model}_{answer_column}_prompt)"
    )
    args = parser.parse_args()

    input_csv = args.input_csv
    output_csv = args.output_csv    
    image_dir = args.image_dir
    question_column = args.question_column
    choices_column = args.choices_column
    answer_gold_column = args.answer_gold_column
    model = args.model
    assert model in SUPPORTED_MODELS, f"Model must be one of: {', '.join(SUPPORTED_MODELS)}"
    # Default answer column/prompt column naming
    if args.answer_column:
        answer_column = args.answer_column
    else:
        answer_column = f"{model}_answer_{question_column}"
    
    prompt_column = args.prompt_column or f"{model}_{answer_column}_prompt"

    assert input_csv != output_csv, "Input and output CSV paths cannot be the same"
    main(
        input_csv,
        output_csv,
        answer_column,
        image_dir,
        model=model,
        prompt_column=prompt_column,
        calc_acc_only=args.calculate_accuracy,
        question_column=question_column,
        choices_column=choices_column,
        answer_gold_column=answer_gold_column,
    )