#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge shard xlsx results for use by evaluate.py.

Features:
- Per benchmark/model_dir/strategy/round processing
- Merges one shard group (explicit timestamp or auto-select latest complete set)
- Cleanup shard xlsx / temp files after successful merge
- Prune duplicate run_info files, keeping only one representative

Usage (run from Bench/script/):
    python merge_shards.py --strategy 1 --benchmark pku --model_dir Qwen2.5-Coder-7B-Instruct
    python merge_shards.py --strategy 0 --benchmark rmc --model_dir Meta-Llama-3-8B-Instruct --cleanup --prune-run-info
"""

import argparse
import os
import glob
import re
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_DIR = SCRIPT_DIR.parent

# Characters illegal in XML 1.0, including the C0/C1 control ranges
# that openpyxl can't parse (U+0000–U+001F minus \t \n \r, U+007F–U+009F,
# surrogates U+D800–U+DFFF, noncharacters U+FFFE/U+FFFF and friends).
_XML_ILLEGAL_RE = re.compile(
    "[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x84\x86-\x9F"
    "\uD800-\uDFFF﷐-﷯￾￿"
    "\U0001FFFE\U0001FFFF\U0002FFFE\U0002FFFF"
    "\U0003FFFE\U0003FFFF\U0004FFFE\U0004FFFF"
    "\U0005FFFE\U0005FFFF\U0006FFFE\U0006FFFF"
    "\U0007FFFE\U0007FFFF\U0008FFFE\U0008FFFF"
    "\U0009FFFE\U0009FFFF\U000AFFFE\U000AFFFF"
    "\U000BFFFE\U000BFFFF\U000CFFFE\U000CFFFF"
    "\U000DFFFE\U000DFFFF\U000EFFFE\U000EFFFF"
    "\U000FFFFE\U000FFFFF\U0010FFFE\U0010FFFF]"
)


def _try_read_xlsx(path: Path) -> pd.DataFrame:
    """Read an xlsx file; on XML parse failure, repair and retry once."""
    try:
        return pd.read_excel(path)
    except Exception as exc:
        err_text = str(exc)
        if "not well-formed" not in err_text and "ParseError" not in err_text:
            raise
        return _repair_and_read(path)


def _repair_and_read(path: Path) -> pd.DataFrame:
    """Unzip the xlsx, strip XML-illegal chars from every XML part, re-pack, read."""
    repaired_path = Path(str(path).rstrip(".xlsx") + ".repaired.xlsx")

    with zipfile.ZipFile(path, "r") as zin:
        members = {}
        for name in zin.namelist():
            members[name] = zin.read(name)

    for name in list(members.keys()):
        if name.endswith(".xml") or name.endswith(".rels"):
            text = members[name].decode("utf-8", errors="replace")
            cleaned = _XML_ILLEGAL_RE.sub("", text)
            members[name] = cleaned.encode("utf-8")

    with zipfile.ZipFile(repaired_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in members.items():
            zout.writestr(name, data)

    df = pd.read_excel(repaired_path)
    repaired_path.unlink()
    return df

SHARD_RE = re.compile(
    r"^res_(?P<strategy>\d+)_py_(?P<timestamp>\d+?)_(?P<model>.+?)_(?P<benchmark>[a-zA-Z0-9_-]+)_round(?P<round>\d+)_done_shard(?P<shard_id>\d+)of(?P<num_shards>\d+)\.xlsx$"
)
RUN_INFO_RE_TEMPLATE = r"^run_info_{model}_{benchmark}_py_s{strategy}_(?P<timestamp>\d+)_round{run_round}_done\.json$"
TEMP_FILE_RE_TEMPLATE = r"^res_{strategy}_py_\d+_{model}_{benchmark}_round{run_round}_temp_shard\d+of\d+\.xlsx$"
CHECKPOINT_RE_TEMPLATE = r"^res_{strategy}_py_\d+_{model}_{benchmark}_round{run_round}_temp_shard\d+of\d+\.checkpoint\.json$"
SHARD_JSON_RE_TEMPLATE = r"^res_{strategy}_py_\d+_{model}_{benchmark}_round{run_round}_done_shard\d+of\d+\.json$"
CKPT_JSON_RE_TEMPLATE = r"^res_{strategy}_py_\d+_{model}_{benchmark}_round{run_round}_temp_shard\d+of\d+\.xlsx\.ckpt\.json$"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", type=int, required=True)
    ap.add_argument("--benchmark", type=str, required=True)
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--run_round", type=int, default=1)
    ap.add_argument("--timestamp", type=str, default=None, help="Only merge shards for this timestamp group.")
    ap.add_argument("--cleanup", action="store_true", help="Delete merged shard files and shard temp/checkpoint files after success.")
    ap.add_argument("--prune-run-info", action="store_true", help="Keep only one matching run_info json in the target round directory.")
    ap.add_argument("--prefer-latest-run-info", action="store_true", help="When pruning run_info, keep the latest timestamp instead of earliest.")
    return ap.parse_args()


def build_output_root(benchmark: str, model_dir: str, run_round: int) -> Path:
    return BENCH_DIR / "Results" / benchmark / model_dir / f"round{run_round}"


def list_shard_items(output_root: Path, strategy: int, benchmark: str, model_dir: str, run_round: int):
    """
    Find all done_shard files and pick the most recent file for each shard_id.
    All shards from the same run config should have the same num_shards.
    Silently ignores files with inconsistent num_shards.
    """
    pattern = str(output_root / f"res_{strategy}_py_*_{model_dir}_{benchmark}_round{run_round}_done_shard*of*.xlsx")
    shard_files = sorted(glob.glob(pattern))

    # Collect all valid shard files, pick most recent for each shard_id
    latest_by_shard: dict = {}
    num_shards_val = None

    for file_path in shard_files:
        name = Path(file_path).name
        match = SHARD_RE.match(name)
        if not match:
            print(f"[Warn] Skip unrecognized shard file name: {name}")
            continue
        sid = int(match.group("shard_id"))
        nshards = int(match.group("num_shards"))
        mtime = os.path.getmtime(file_path)

        if num_shards_val is None:
            num_shards_val = nshards
        elif nshards != num_shards_val:
            # Different num_shards means different run config — skip
            continue

        if sid not in latest_by_shard or mtime > latest_by_shard[sid]["mtime"]:
            latest_by_shard[sid] = {
                "path": Path(file_path),
                "timestamp": match.group("timestamp"),
                "shard_id": sid,
                "num_shards": nshards,
                "mtime": mtime,
            }

    if num_shards_val is None:
        return [], False

    items = sorted(latest_by_shard.values(), key=lambda x: x["shard_id"])
    shard_ids = [item["shard_id"] for item in items]
    complete = len(items) == num_shards_val and shard_ids == list(range(num_shards_val))
    return items, complete


def choose_shards(items, complete, explicit_timestamp=None):
    if not items:
        return None, []

    if explicit_timestamp is not None:
        subset = [i for i in items if i["timestamp"] == explicit_timestamp]
        return explicit_timestamp, subset

    if not complete:
        n = len(items)
        expected = items[0]["num_shards"] if items else 0
        shard_ids = [i["shard_id"] for i in items]
        print(
            f"[Warn] Incomplete shard set: found shard ids {shard_ids}, expected 0..{expected-1}"
        )
    return items[0]["timestamp"], items


def merge_group(output_root: Path, strategy: int, benchmark: str, model_dir: str, run_round: int, items):
    if not items:
        raise FileNotFoundError("No shard files to merge")

    num_shards = items[0]["num_shards"]
    shard_ids = [item["shard_id"] for item in items]
    if len(items) != num_shards or shard_ids != list(range(num_shards)):
        raise RuntimeError(
            f"Shard set is incomplete: got shard ids {shard_ids}, expected 0..{num_shards-1}"
        )

    print(f"Merging {len(items)}/{num_shards} shards:")
    dfs = []
    for item in items:
        df = _try_read_xlsx(item["path"])
        dfs.append(df)
        print(f"  {item['path'].name}: {len(df)} rows")

    merged = pd.concat(dfs, ignore_index=True)
    # Use current timestamp for the merged output file
    out_ts = time.strftime("%Y%m%d%H%M%S")
    out_name = f"res_{strategy}_py_{out_ts}_{model_dir}_{benchmark}_round{run_round}_done.xlsx"
    out_path = output_root / out_name
    merged.to_excel(out_path, index=False)
    print(f"[OK] Merged {len(merged)} rows -> {out_path}")
    return out_path


def cleanup_shard_artifacts(output_root: Path, strategy: int, benchmark: str, model_dir: str, run_round: int, items):
    for item in items:
        shard_path = item["path"]
        if shard_path.exists():
            shard_path.unlink()
            print(f"  Deleted shard: {shard_path.name}")

    temp_re = re.compile(TEMP_FILE_RE_TEMPLATE.format(
        strategy=strategy,
        model=re.escape(model_dir),
        benchmark=re.escape(benchmark),
        run_round=run_round,
    ))
    checkpoint_re = re.compile(CHECKPOINT_RE_TEMPLATE.format(
        strategy=strategy,
        model=re.escape(model_dir),
        benchmark=re.escape(benchmark),
        run_round=run_round,
    ))

    for candidate in output_root.iterdir():
        name = candidate.name
        if temp_re.match(name) or checkpoint_re.match(name):
            candidate.unlink()
            print(f"  Deleted temp/checkpoint: {name}")

    # Also clean up shard JSON sidecar files
    shard_json_re = re.compile(SHARD_JSON_RE_TEMPLATE.format(
        strategy=strategy,
        model=re.escape(model_dir),
        benchmark=re.escape(benchmark),
        run_round=run_round,
    ))
    ckpt_json_re = re.compile(CKPT_JSON_RE_TEMPLATE.format(
        strategy=strategy,
        model=re.escape(model_dir),
        benchmark=re.escape(benchmark),
        run_round=run_round,
    ))
    for candidate in output_root.iterdir():
        name = candidate.name
        if shard_json_re.match(name) or ckpt_json_re.match(name):
            candidate.unlink()
            print(f"  Deleted shard json: {name}")


def prune_run_info(output_root: Path, model_dir: str, benchmark: str, strategy: int, run_round: int, keep_latest: bool):
    run_info_re = re.compile(RUN_INFO_RE_TEMPLATE.format(
        model=re.escape(model_dir),
        benchmark=re.escape(benchmark),
        strategy=strategy,
        run_round=run_round,
    ))

    matched = []
    for candidate in output_root.iterdir():
        match = run_info_re.match(candidate.name)
        if match:
            matched.append((match.group("timestamp"), candidate))

    if len(matched) <= 1:
        if matched:
            print(f"[Info] Single run_info retained: {matched[0][1].name}")
        else:
            print("[Info] No matching run_info files to prune.")
        return

    matched.sort(key=lambda x: x[0])
    keep = matched[-1] if keep_latest else matched[0]
    print(f"[Info] Keeping run_info: {keep[1].name}")
    for timestamp, path in matched:
        if path == keep[1]:
            continue
        path.unlink()
        print(f"  Deleted duplicate run_info: {path.name}")


def main():
    args = parse_args()
    output_root = build_output_root(args.benchmark, args.model_dir, args.run_round)
    if not output_root.exists():
        raise FileNotFoundError(f"Output directory not found: {output_root}")

    items, complete = list_shard_items(output_root, args.strategy, args.benchmark, args.model_dir, args.run_round)
    _timestamp, items = choose_shards(items, complete, args.timestamp)

    if items:
        if not complete:
            raise RuntimeError(
                f"Shard set incomplete: got {len(items)}/{items[0]['num_shards']} shards. "
                f"Missing shard_ids: {set(range(items[0]['num_shards'])) - {i['shard_id'] for i in items}}"
            )
        merge_group(output_root, args.strategy, args.benchmark, args.model_dir, args.run_round, items)
        if args.cleanup:
            cleanup_shard_artifacts(output_root, args.strategy, args.benchmark, args.model_dir, args.run_round, items)
    else:
        print("[Info] No shard xlsx selected for merge.")

    if args.prune_run_info:
        prune_run_info(output_root, args.model_dir, args.benchmark, args.strategy, args.run_round, args.prefer_latest_run_info)


if __name__ == "__main__":
    main()
