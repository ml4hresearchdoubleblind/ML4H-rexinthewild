#!/usr/bin/env python3
"""
Run one model over the benchmark and write a provenance sidecar next to the
results, so a results CSV is never separated from the exact configuration that
produced it.

  python run_with_provenance.py MODEL INPUT_CSV IMAGE_DIR [--outdir results]
                                [--limit N] [--tag NAME]

Produces, in --outdir:
  <stem>.csv        model answers (written by run_inference.py)
  <stem>.meta.json  model id, every decoding parameter, API version, prompts,
                    dataset + script checksums, library versions, timings
  <stem>.log        full stdout/stderr of the run
"""
import argparse, ast, csv, hashlib, json, os, platform, re, socket, subprocess, sys, time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "run_inference.py")
CONDA_PY = os.environ.get("REXIN_PYTHON", sys.executable)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def static_read(script):
    """Pull MODEL_PARAMS and the prompts without importing (avoids side effects)."""
    src = open(script, encoding="utf-8").read()
    params = ast.literal_eval(re.search(r"MODEL_PARAMS = (\{.*?\n\})\n", src, re.S).group(1))
    sysp = re.search(r'SYSTEM_PROMPT = """(.*?)"""', src, re.S).group(1)
    userp = ast.literal_eval("(" + re.search(r"USER_PROMPT = \((.*?)\n\)", src, re.S).group(1) + ")")
    resolved = re.findall(r'model="((?:claude|gemini)-[^"]+)"', src)
    return params, sysp, userp, resolved


def libver(py, mod):
    try:
        out = subprocess.run([py, "-c",
              f"import importlib;m=importlib.import_module('{mod}');print(getattr(m,'__version__','?'))"],
              capture_output=True, text=True, timeout=60)
        return out.stdout.strip() or None
    except Exception:
        return None


def rowcount(path):
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return sum(1 for _ in csv.DictReader(f))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("input_csv")
    ap.add_argument("image_dir")
    ap.add_argument("--outdir", default=os.path.join(HERE, "results"))
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N rows (smoke test)")
    ap.add_argument("--tag", default=None, help="extra label in the filename")
    ap.add_argument("--stem", default=None,
                    help="fixed output stem (no timestamp). REQUIRED for resumable "
                         "long runs: re-running the same command appends to the same "
                         "CSV and skips questions already answered.")
    ap.add_argument("--question_column", default="question")
    ap.add_argument("--choices_column", default="choices")
    ap.add_argument("--answer_gold_column", default="answer")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = a.stem or f"{a.model}_{a.tag + '_' if a.tag else ''}{'smoke%d_' % a.limit if a.limit else ''}{stamp}"
    out_csv = os.path.join(a.outdir, stem + ".csv")
    out_meta = os.path.join(a.outdir, stem + ".meta.json")
    out_log = os.path.join(a.outdir, stem + ".log")

    src_csv = a.input_csv
    if a.limit:                       # truncated copy for smoke tests
        tmpdir = os.path.join(a.outdir, "tmp"); os.makedirs(tmpdir, exist_ok=True)
        src_csv = os.path.join(tmpdir, stem + ".input.csv")
        with open(a.input_csv, newline="", encoding="utf-8") as f:
            rd = csv.DictReader(f); rows = [next(rd) for _ in range(a.limit)]
            with open(src_csv, "w", newline="", encoding="utf-8") as g:
                w = csv.DictWriter(g, fieldnames=rd.fieldnames)
                w.writeheader(); w.writerows(rows)

    params, sysp, userp, resolved_ids = static_read(SCRIPT)
    answer_col = f"{a.model}_answer"

    cmd = [CONDA_PY if os.path.exists(CONDA_PY) else sys.executable, SCRIPT,
           src_csv, out_csv, a.model, a.image_dir,
           "--question_column", a.question_column,
           "--choices_column", a.choices_column,
           "--answer_gold_column", a.answer_gold_column,
           "--answer_column", answer_col]

    meta = {
        "run": {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "command": cmd,
        },
        "model": {
            "key": a.model,
            "decoding_parameters": params.get(a.model),
            "resolved_model_id": (params.get(a.model) or {}).get("model_id"),
            "model_ids_in_script": sorted(set(resolved_ids)),
            "azure_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT") or "(default in script)",
            "azure_api_version": os.getenv("AZURE_OPENAI_API_VERSION") or "(default in script)",
            "azure_endpoint_host": (os.getenv("AZURE_OPENAI_ENDPOINT") or "").split("//")[-1].split("/")[0] or None,
            "google_use_vertexai": os.getenv("GOOGLE_GENAI_USE_VERTEXAI"),
            "google_cloud_project": os.getenv("GOOGLE_CLOUD_PROJECT"),
            "google_cloud_location": os.getenv("GOOGLE_CLOUD_LOCATION"),
            "google_auth": ("ADC (application_default_credentials.json)"
                            if (os.getenv("GOOGLE_GENAI_USE_VERTEXAI") or "").lower() in ("1","true","yes")
                            else ("GOOGLE_API_KEY" if os.getenv("GOOGLE_API_KEY") else None)),
            "gemini_sleep_seconds": os.getenv("GEMINI_SLEEP_SECONDS", "5"),
        },
        "prompts": {"system": sysp, "user": userp},
        "dataset": {
            "path": os.path.abspath(a.input_csv),
            "sha256": sha256(a.input_csv),
            "rows": rowcount(a.input_csv),
            "rows_run": a.limit or rowcount(a.input_csv),
            "question_column": a.question_column,
            "choices_column": a.choices_column,
            "answer_gold_column": a.answer_gold_column,
        },
        "code": {
            "run_inference.py_sha256": sha256(SCRIPT),
            "run_with_provenance.py_sha256": sha256(os.path.abspath(__file__)),
        },
        "libraries": {m: libver(cmd[0], m) for m in
                      ["google.genai", "openai", "anthropic", "transformers", "torch"]},
    }
    # Record the model snapshot the endpoint actually serves -- a deployment name
    # can be repointed, so the deployment alone is not a version record.
    _mid = (params.get(a.model) or {}).get("model_id")
    if a.model in ("gemini", "gemini37flash"):
        probe = ("import os;from google import genai;from google.genai import types;"
                 "c=genai.Client(vertexai=True,project=os.environ['GOOGLE_CLOUD_PROJECT'],"
                 "location=os.environ['GOOGLE_CLOUD_LOCATION'],"
                 "http_options=types.HttpOptions(timeout=60000));"
                 f"r=c.models.generate_content(model='{_mid}',contents='hi');"
                 "print(getattr(r,'model_version',None) or '')")
        try:
            out = subprocess.run([cmd[0], "-c", probe], capture_output=True, text=True, timeout=180)
            meta["model"]["served_model_snapshot"] = out.stdout.strip() or None
        except Exception:
            meta["model"]["served_model_snapshot"] = None
    if a.model in ("claude-opus5", "claude-opus5-max") and os.getenv("CLAUDE_API_KEY"):
        probe = ("import os,anthropic;"
                 "c=anthropic.Anthropic(api_key=os.environ['CLAUDE_API_KEY'],"
                 "default_headers={'anthropic-workspace-id':os.environ.get('ANTHROPIC_WORKSPACE_ID','')});"
                 f"print(c.messages.create(model='{_mid}',max_tokens=16,"
                 "messages=[{'role':'user','content':'hi'}]).model)")
        try:
            out = subprocess.run([cmd[0], "-c", probe], capture_output=True, text=True, timeout=180)
            meta["model"]["served_model_snapshot"] = out.stdout.strip() or None
        except Exception:
            meta["model"]["served_model_snapshot"] = None
    if a.model in ("gpt", "gpt-xhigh", "gpt6astra", "gpt6astra-xhigh") and os.getenv("AZURE_OPENAI_API_KEY"):
        probe = ("import os,openai;"
                 "c=openai.AzureOpenAI(api_key=os.environ['AZURE_OPENAI_API_KEY'],"
                 "azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],"
                 "api_version=os.environ.get('AZURE_OPENAI_API_VERSION','2024-12-01-preview'),timeout=45);"
                 "print(c.chat.completions.create(model=os.environ['AZURE_OPENAI_DEPLOYMENT'],"
                 "messages=[{'role':'user','content':'hi'}]).model)")
        try:
            out = subprocess.run([cmd[0], "-c", probe], capture_output=True, text=True, timeout=120)
            meta["model"]["served_model_snapshot"] = out.stdout.strip() or None
        except Exception:
            meta["model"]["served_model_snapshot"] = None

    try:
        meta["code"]["git_commit"] = subprocess.run(
            ["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip() or None
    except Exception:
        meta["code"]["git_commit"] = None

    print(f"model      : {a.model}  {params.get(a.model)}")
    print(f"dataset    : {meta['dataset']['rows_run']} rows from {os.path.basename(a.input_csv)}")
    print(f"results -> {out_csv}\nmeta    -> {out_meta}\nlog     -> {out_log}\n", flush=True)

    meta["run"]["status"] = "running"
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    t0 = time.time()
    with open(out_log, "a", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0

    meta["run"]["finished_utc"] = datetime.now(timezone.utc).isoformat()
    meta["run"]["duration_seconds"] = round(dt, 1)
    meta["run"]["exit_code"] = proc.returncode
    meta["run"]["status"] = "completed" if proc.returncode == 0 else "failed"
    meta["results"] = {
        "path": os.path.abspath(out_csv),
        "answer_column": answer_col,
        "rows_written": rowcount(out_csv),
        "sha256": sha256(out_csv) if os.path.exists(out_csv) else None,
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nexit {proc.returncode} in {dt:.0f}s; wrote {meta['results']['rows_written']} rows")
    if proc.returncode != 0:
        print(f"FAILED -- see {out_log}")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
