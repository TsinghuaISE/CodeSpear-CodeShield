"""
Main entry point — iterate over models × benchmarks × methods, call APIs,
and persist results to APIModels/Results/.

Usage:
    python run.py                                          # run everything
    python run.py --models gpt-5-mini --benchmarks mal     # subset
    python run.py --methods Vanilla Ours                   # specific methods
    python run.py --rounds 1                               # round number(s)
"""

import argparse
import json
import os
import sys
import time
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd

import config
from clients import (
    call_openai_normal,
    call_openai_ours,
    call_fireworks_normal,
    call_fireworks_ours,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_print_lock = threading.Lock()
_MAX_EXCEL_CELL_CHARS = 32767
_ILLEGAL_XLSX_CHAR_RE = __import__("re").compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")


def _ts_print(*args, **kwargs):
    """Thread-safe print with timestamp prefix."""
    ts = time.strftime("%H:%M:%S")
    with _print_lock:
        print(f"[{ts}]", *args, **kwargs)


def _sanitize(text, max_chars=_MAX_EXCEL_CELL_CHARS):
    s = str(text)
    s = _ILLEGAL_XLSX_CHAR_RE.sub("", s)
    return s[:max_chars]


def _safe_to_excel(df, path):
    """Write Excel, sanitizing cells on failure."""
    try:
        df.to_excel(path, index=False)
    except Exception:
        for col in df.select_dtypes(include=["object", "string"]).columns:
            df[col] = df[col].apply(lambda v: _sanitize(v) if isinstance(v, str) else v)
        df.to_excel(path, index=False)


# ---- file-name builder ----------------------------------------------------

def _method_slug(method: str) -> str:
    """'Low-Resource Language' -> 'Low-Resource_Language' for safe filenames."""
    return method.replace(" ", "_")


def _build_filename(method: str, model: str, benchmark: str,
                    run_round: int, status: str, ext: str,
                    timestamp: str = None) -> str:
    parts = [_method_slug(method), model, benchmark]
    parts.append(f"round{run_round}")
    if timestamp:
        parts.append(timestamp)
    parts.append(status)
    return "_".join(parts) + f".{ext}"


# ---- prompt-file resolution -----------------------------------------------

def _resolve_prompt_file(benchmark: str, method: str) -> str:
    """Return the xlsx path for a given (benchmark, method)."""
    if method in (config.VANILLA, config.OURS):
        return config.VANILLA_PROMPT_FILES[benchmark]
    return config.BASELINE_PROMPT_FILES[benchmark][method]


# ---- API call dispatch ----------------------------------------------------

def _is_openai_model(model: str) -> bool:
    return model in config.OPENAI_MODELS


def _call_api(prompt: str, model: str, method: str) -> str:
    """Route to the right API function based on model provider and method."""
    ours = (method == config.OURS)

    if _is_openai_model(model):
        if ours:
            return call_openai_ours(prompt, model)
        else:
            return call_openai_normal(prompt, model)
    else:
        # Fireworks model
        if ours:
            return call_fireworks_ours(prompt, model)
        else:
            return call_fireworks_normal(prompt, model)

# ---- single run -----------------------------------------------------------

def _reset_error_rows(df):
    """Clear done flag + response for rows whose response starts with 'Error:'."""
    if "response" not in df.columns:
        return 0
    mask = df["response"].fillna("").astype(str).str.startswith("Error:")
    n = int(mask.sum())
    if n > 0:
        df.loc[mask, "done"] = False
    return n


import glob as _glob


def _find_existing_done(method, model, benchmark, run_round):
    """Return the path of the most recent done file for this task, or None."""
    pattern = os.path.join(
        config.OUTPUT_DIR,
        _build_filename(method, model, benchmark, run_round, "done", "xlsx",
                        timestamp="*").replace("round1_*_done", "round1_??????????????_done"),
    )
    # Simpler: glob with wildcard
    prefix = f"{_method_slug(method)}_{model}_{benchmark}_round{run_round}_"
    candidates = _glob.glob(os.path.join(config.OUTPUT_DIR, prefix + "*_done.xlsx"))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def run_one(model: str, benchmark: str, method: str, run_round: int = 1):
    """Run one (model, benchmark, method, round) combination."""

    prompt_file = _resolve_prompt_file(benchmark, method)
    if not os.path.exists(prompt_file):
        _ts_print(f"[Skip] prompt file not found: {prompt_file}")
        return

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    slug = _method_slug(method)
    temp_file = os.path.join(
        config.OUTPUT_DIR,
        _build_filename(method, model, benchmark, run_round, "temp", "xlsx"),
    )

    # --- load: temp > done (auto-convert to temp) > fresh ---
    if os.path.exists(temp_file):
        _ts_print(f"[{model}|{benchmark}|{method}] Resuming from temp file …")
        df = pd.read_excel(temp_file, dtype=str)
    else:
        done_file = _find_existing_done(method, model, benchmark, run_round)
        if done_file:
            df = pd.read_excel(done_file, dtype=str)
            _ts_print(f"[{model}|{benchmark}|{method}] "
                      f"Loading from done, converting to temp …")
        else:
            df = pd.read_excel(prompt_file, dtype=str)
            _ts_print(f"[{model}|{benchmark}|{method}] Starting fresh …")

    # ensure columns
    if "done" not in df.columns:
        df["done"] = False
    if "response" not in df.columns:
        df["response"] = ""
    df["response"] = df["response"].fillna("")

    # Normalize "done" column: reading with dtype=str turns booleans into
    # the strings "True"/"False", and "False" as a non-empty string is truthy
    # under .astype(bool).  Map back to proper bools so the start_idx logic
    # (and _reset_error_rows) work correctly.
    if df["done"].dtype == object:
        df["done"] = df["done"].astype(str).str.strip().str.lower().map(
            {"true": True, "false": False})

    # always reset error rows so they get retried
    n_err = _reset_error_rows(df)
    if n_err > 0:
        _ts_print(f"[{model}|{benchmark}|{method}] "
                  f"Retrying {n_err} error row(s) …")

    # find first incomplete row
    done_series = df["done"].fillna(False).astype(bool)
    incomplete = done_series[~done_series]
    start_idx = int(incomplete.index[0]) if len(incomplete) > 0 else len(df)

    total = len(df)
    _ts_print(f"[{model}|{benchmark}|{method}] "
              f"round={run_round} | {total} prompts | start at {start_idx}")

    start_time = time.perf_counter()
    errors = []

    for idx in range(start_idx, total):
        prompt_text = str(df.at[idx, "prompt"])

        final_prompt = prompt_text
        try:
            response = _call_api(final_prompt, model, method)
        except Exception as exc:
            _ts_print(f"[{model}|{benchmark}|{method}] "
                      f"[Error] row {idx}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            response = f"Error: {exc}"
            errors.append(idx)

        df.at[idx, "response"] = _sanitize(response)
        df.at[idx, "done"] = True

        # save after every single prompt so progress is never lost
        _safe_to_excel(df, temp_file)

        # progress line
        elapsed = int(time.perf_counter() - start_time)
        eta = ""
        if idx >= start_idx + 1:
            avg = elapsed / max(idx - start_idx, 1)
            remaining = int(avg * (total - idx - 1))
            m, s = divmod(remaining, 60)
            eta = f" | ETA {m:02d}m{s:02d}s"
        _ts_print(f"[{model}|{benchmark}|{method}] "
                  f"[{idx + 1}/{total}] {(idx + 1) / total * 100:.1f}%  "
                  f"elapsed {elapsed}s{eta}")

    print()  # newline after progress

    # --- save final outputs ---
    timestamp = time.strftime("%Y%m%d%H%M%S")
    done_file = os.path.join(
        config.OUTPUT_DIR,
        _build_filename(method, model, benchmark, run_round, "done", "xlsx",
                        timestamp=timestamp),
    )
    json_file = os.path.join(
        config.OUTPUT_DIR,
        _build_filename(method, model, benchmark, run_round, "done", "json",
                        timestamp=timestamp),
    )

    _safe_to_excel(df, done_file)
    df.to_json(json_file, orient="records")

    # run info
    info = {
        "model": model,
        "benchmark": benchmark,
        "method": method,
        "run_round": run_round,
        "timestamp": timestamp,
        "num_prompts": total,
        "errors": errors,
        "time_consuming_s": int(time.perf_counter() - start_time),
        "sampling": config.SAMPLING,
    }
    info_file = os.path.join(
        config.OUTPUT_DIR,
        _build_filename(method, model, benchmark, run_round, "info", "json",
                        timestamp=timestamp),
    )
    with open(info_file, "w") as f:
        json.dump(info, f, indent=2)

    # clean up temp
    if os.path.exists(temp_file):
        os.remove(temp_file)

    _ts_print(f"[{model}|{benchmark}|{method}] Done -> {done_file}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="API Model Evaluation")
    parser.add_argument("--models", nargs="+",
                        default=config.OPENAI_MODELS + config.FIREWORKS_MODELS,
                        help="Models to evaluate")
    parser.add_argument("--benchmarks", nargs="+",
                        default=config.ALL_BENCHMARKS,
                        help="Benchmarks to evaluate (mal, rmc)")
    parser.add_argument("--methods", nargs="+",
                        default=config.ALL_METHODS,
                        help="Methods to evaluate")
    parser.add_argument("--rounds", type=int, nargs="+", default=[1],
                        help="Run round numbers (default: 1)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers (default: 8)")
    parser.add_argument("--check_errors", action="store_true",
                        help="Scan for error rows and print summary, then exit (no run)")
    args = parser.parse_args()

    # Build flat list of tasks
    tasks = []
    for model in args.models:
        for benchmark in args.benchmarks:
            for method in args.methods:
                for rnd in args.rounds:
                    tasks.append((model, benchmark, method, rnd))

    print("=== API Model Evaluation ===")
    print(f"Models:     {args.models}")
    print(f"Benchmarks: {args.benchmarks}")
    print(f"Methods:    {args.methods}")
    print(f"Rounds:     {args.rounds}")
    print(f"Workers:    {args.workers}")
    print(f"Tasks:      {len(tasks)}")
    print(f"Output:     {config.OUTPUT_DIR}")
    print()

    # --check_errors: scan and report, then exit
    if args.check_errors:
        total_errors = 0
        print(f"{'Model':15s} {'Bench':6s} {'Method':20s} {'Rnd':5s} {'Errors':>7s}  {'Source'}")
        print("-" * 80)
        for model, benchmark, method, rnd in tasks:
            temp_file = os.path.join(
                config.OUTPUT_DIR,
                _build_filename(method, model, benchmark, rnd, "temp", "xlsx"),
            )
            done_file = _find_existing_done(method, model, benchmark, rnd)

            source, df = None, None
            if os.path.exists(temp_file):
                source, df = "temp", pd.read_excel(temp_file, dtype=str)
            elif done_file:
                source, df = f"done ({os.path.basename(done_file)[:40]})", pd.read_excel(done_file, dtype=str)

            if df is not None and "response" in df.columns:
                n = int(df["response"].fillna("").astype(str).str.startswith("Error:").sum())
            else:
                n = 0
            total_errors += n

            marker = " !" if n > 0 else ""
            print(f"{model:15s} {benchmark:6s} {method:20s} {rnd:5d} {n:7d}  {source or '—'}{marker}")

        print("-" * 80)
        print(f"Total error rows: {total_errors}")
        if total_errors > 0:
            print("\nRun with --retry_errors to reset and retry them.")
        return

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_one, model, benchmark, method, rnd): (model, benchmark, method, rnd)
            for (model, benchmark, method, rnd) in tasks
        }

        for future in as_completed(futures):
            model, benchmark, method, rnd = futures[future]
            try:
                future.result()
            except Exception as exc:
                _ts_print(f"[FATAL] {model}|{benchmark}|{method}|round{rnd}: "
                          f"{type(exc).__name__}: {exc}")
                traceback.print_exc()

    print("\nAll done.")


if __name__ == "__main__":
    main()
