"""
Flow:
  1. From eval_result xlsx, find verdict==BAD responses (model failed to defend)
  2. k = min(--k, n_bad)
  3. From SFT source (OCI), sample k*j Python records for DPO (no malicious keywords)
  4. From SFT source, sample S*k*2*j records for SFT (all languages, no malicious, no overlap with DPO)
  5. For each BAD prompt i, for jj in 1..j:
       Stage 1: chosen  = rejection_i (NL refusal)
                rejected = sft_code_(i,jj)
       Stage 2: chosen  = sft_code_(i,jj)
                rejected = S1_response_i (model's malicious code)
  6. Shuffle all 2*j*k DPO pairs
  7. Collect S*k*2*j SFT pairs (problem->solution, unmodified)

Format: LLaMA-Factory ShareGPT
  - DPO: conversations + chosen + rejected
  - SFT: conversations (human/gpt turns)

Usage:
  python build_training_data.py \
    --eval_result  ../Attack/LocalModels/Results/rmc/<model_dir>/round1/<result>.xlsx \
    --rejections   ../Attack/LocalModels/Results/rmc/<model_dir>/round1/<rejections>.xlsx \
    --sft_source   OCI \
    --sft_data     ../data/OpenCodeInstruct_sft_57500.json \
    --output_dpo   data/qwencoder7b/dpo.json \
    --output_sft   data1/qwencoder7b/sft.json \
    --k 1600 --j 5 --SFT 2.5 --seed 42
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

CODEBLOCK_RE = re.compile(r"```(?:[a-zA-Z+#]+)?\n(.*?)\n```", re.DOTALL)
CODEBLOCK_LANG_RE = re.compile(r"```([a-zA-Z+#]+)\n", re.IGNORECASE)

LANG_ALIASES = {
    "py": "python", "python": "python",
    "java": "java", "cpp": "cpp", "c++": "cpp", "cxx": "cpp",
    "c": "c", "js": "javascript", "javascript": "javascript",
    "ts": "typescript", "typescript": "typescript",
    "go": "go", "golang": "go", "rust": "rust",
    "csharp": "csharp", "c#": "csharp", "cs": "csharp", "php": "php",
}


def load_malicious_keywords(path: str) -> List[str]:
    keywords = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            kw = line.strip().lower()
            if kw:
                keywords.append(kw)
    print(f"[keywords] {len(keywords)} malicious keywords loaded", file=sys.stderr)
    return keywords


def has_malicious_content(text: str, keywords: List[str]) -> bool:
    text_lower = text.lower()
    for kw in keywords:
        if kw in text_lower:
            return True
    return False


def is_python(rec: Dict) -> bool:
    """Check if a normalized record's solution is Python code."""
    sol = rec.get("solution", "")
    m = CODEBLOCK_LANG_RE.search(sol)
    if m:
        return LANG_ALIASES.get(m.group(1).lower()) == "python"
    sol_stripped = sol.strip()
    return (
        sol_stripped.startswith("def ")
        or sol_stripped.startswith("import ")
        or sol_stripped.startswith("from ")
        or sol_stripped.startswith("class ")
        or "print(" in sol_stripped[:200]
        or "= [" in sol_stripped[:200]
    )


def extract_code_body(solution: str) -> str:
    """Extract the first code block body (strip markdown fences)."""
    m = CODEBLOCK_RE.search(solution)
    if m:
        return m.group(1)
    return solution.strip()


def format_dpo_code(code_body: str) -> str:
    """Wrap code with S1-aligned prefix: ```python\nimport os\n<body>\n```"""
    return f"```python\nimport os\n{code_body}\n```"


def load_magicoder_pool(path: str, keywords: List[str], python_only: bool = True) -> List[Dict]:
    """Load magicoder_normalized.jsonl, filter malicious content, optionally Python-only.
    Returns normalized records: {"problem": ..., "solution": ...}"""
    pool = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            if python_only and not is_python(rec):
                continue

            combined = rec.get("problem", "") + " " + rec.get("solution", "")
            if has_malicious_content(combined, keywords):
                continue

            pool.append({"problem": rec.get("problem", ""), "solution": rec.get("solution", "")})

    tag = "python " if python_only else ""
    print(f"[magicoder] {len(pool)} eligible {tag}records (no malicious)", file=sys.stderr)
    return pool


def load_oci_pool(path: str, keywords: List[str], python_only: bool = True) -> List[Dict]:
    """Load OCI ShareGPT JSON, filter malicious content, optionally Python-only.
    Returns normalized records: {"problem": ..., "solution": ...}"""
    pool = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for rec in data:
        convs = rec.get("conversations", [])
        problem = ""
        solution = ""
        for turn in convs:
            if turn.get("from") == "human":
                problem = turn.get("value", "")
            elif turn.get("from") == "gpt":
                solution = turn.get("value", "")

        if not problem or not solution:
            continue

        if python_only and not is_python({"solution": solution}):
            continue

        combined = problem + " " + solution
        if has_malicious_content(combined, keywords):
            continue

        pool.append({"problem": problem, "solution": solution})

    tag = "python " if python_only else ""
    print(f"[OCI] {len(pool)} eligible {tag}records (no malicious)", file=sys.stderr)
    return pool


def load_sft_pool(path: str, source: str, keywords: List[str], python_only: bool = True) -> List[Dict]:
    """Load SFT source pool, returning normalized records: {"problem": ..., "solution": ...}"""
    if source == "OCI":
        return load_oci_pool(path, keywords, python_only)
    else:
        return load_magicoder_pool(path, keywords, python_only)


def make_sft_record(rec: Dict) -> Dict:
    """Convert a normalized record to LLaMA-Factory ShareGPT SFT format."""
    return {
        "conversations": [
            {"from": "human", "value": rec["problem"]},
            {"from": "gpt", "value": rec["solution"]},
        ],
        "_source": "method2_sft",
    }


def find_column(df: pd.DataFrame, *names: str) -> Optional[str]:
    for name in names:
        for c in df.columns:
            if c.lower() == name.lower():
                return c
    return None


def build_method2_data(
    eval_result_path: str,
    rejections_path: str,
    sft_data_path: str,
    sft_source: str,
    keywords_path: str,
    k: Optional[int],
    seed: int,
    j: int = 10,
    SFT: float = 2.5,
    sft_python_only: bool = False,
    dpo_data_path: Optional[str] = None,
    dpo_source: Optional[str] = None,
) -> Tuple[List[Dict], List[Dict]]:
    # DPO random code defaults to SFT source if not specified
    if dpo_source is None:
        dpo_source = sft_source
    if dpo_data_path is None:
        dpo_data_path = sft_data_path

    rng = random.Random(seed)

    # ---- load keywords ----
    keywords = load_malicious_keywords(keywords_path)

    # ---- load eval results ----
    print(f"[load] eval_result: {eval_result_path}", file=sys.stderr)
    eval_df = pd.read_excel(eval_result_path)
    print(f"  rows={len(eval_df)}", file=sys.stderr)

    if "verdict" not in eval_df.columns:
        print("[FATAL] eval_result has no 'verdict' column", file=sys.stderr)
        sys.exit(1)

    # ---- load rejections ----
    print(f"[load] rejections: {rejections_path}", file=sys.stderr)
    rej_df = pd.read_excel(rejections_path)
    print(f"  rows={len(rej_df)}", file=sys.stderr)

    # ---- resolve columns ----
    prompt_col = find_column(eval_df, "prompt")
    resp_col = find_column(eval_df, "response")
    rej_col = find_column(rej_df, "rejection")

    if not prompt_col or not resp_col:
        print("[FATAL] Cannot find prompt/response column in eval_result", file=sys.stderr)
        sys.exit(1)
    if not rej_col:
        print("[FATAL] Cannot find rejection column in rejections", file=sys.stderr)
        sys.exit(1)

    # ---- filter BAD verdicts ----
    bad_mask = eval_df["verdict"].str.strip().str.upper() == "BAD"
    bad_indices = eval_df[bad_mask].index.tolist()
    n_bad = len(bad_indices)
    print(f"[stats] BAD verdicts: {n_bad}", file=sys.stderr)

    # ---- determine k ----
    if k is not None and k > 0:
        k = min(k, n_bad)
    else:
        k = n_bad
    print(f"[stats] k = {k}", file=sys.stderr)

    if k == 0:
        print("[FATAL] No BAD prompts to process", file=sys.stderr)
        sys.exit(1)

    # ---- load SFT source pools (DPO: python only; SFT: configurable) ----
    dpo_pool = load_sft_pool(dpo_data_path, dpo_source, keywords, python_only=True)
    sft_pool = load_sft_pool(sft_data_path, sft_source, keywords, python_only=sft_python_only)

    dpo_needed = k * j
    dpo_count = k * 2 * j
    sft_count = int(dpo_count * SFT)
    sft_needed = sft_count

    if len(dpo_pool) < dpo_needed:
        print(
            f"[FATAL] Not enough [{dpo_source}] records for DPO: need {dpo_needed}, have {len(dpo_pool)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---- sample DPO records (python only, no replacement) ----
    rng.shuffle(dpo_pool)
    dpo_recs = dpo_pool[:dpo_needed]

    # ---- sample SFT records (no replacement) ----
    if len(sft_pool) < sft_needed:
        print(
            f"[FATAL] Not enough [{sft_source}] records for SFT: need {sft_needed}, have {len(sft_pool)}",
            file=sys.stderr,
        )
        sys.exit(1)
    rng.shuffle(sft_pool)
    sft_recs = sft_pool[:sft_needed]

    # ---- build DPO data ----
    dpo_data: List[Dict] = []
    sft_data: List[Dict] = []

    for i in range(k):
        eval_idx = bad_indices[i]
        pid = eval_df.at[eval_idx, "pid"]

        raw_prompt = str(eval_df.at[eval_idx, prompt_col])
        s1_response = str(eval_df.at[eval_idx, resp_col])

        # Look up rejection by pid
        rej_rows = rej_df[rej_df["pid"] == pid]
        if len(rej_rows) == 0:
            print(f"[WARN] pid={pid} not found in rejections, skipping", file=sys.stderr)
            continue
        rejection = str(rej_rows.iloc[0][rej_col])

        # Skip empty S1 responses
        if s1_response.strip().lower() in ("", "nan", "none"):
            print(f"[WARN] pid={pid} has empty S1 response, skipping", file=sys.stderr)
            continue

        prompt_with_suffix = raw_prompt  # v13+: NO_SUFFIX — align with run_main S1 behavior

        for jj in range(j):
            d_idx = i * j + jj
            d_rec = dpo_recs[d_idx]

            code_body = extract_code_body(d_rec["solution"])
            code_with_prefix = format_dpo_code(code_body)

            # Stage 1: NL refusal > useless code
            dpo_data.append({
                "conversations": [{"from": "human", "value": prompt_with_suffix}],
                "chosen": {"from": "gpt", "value": rejection},
                "rejected": {"from": "gpt", "value": code_with_prefix},
                "_source": "method2_stage1",
                "_pid": int(pid),
                "_pair": jj + 1,
            })

            # Stage 2: useless code > malicious code
            dpo_data.append({
                "conversations": [{"from": "human", "value": prompt_with_suffix}],
                "chosen": {"from": "gpt", "value": code_with_prefix},
                "rejected": {"from": "gpt", "value": s1_response},
                "_source": "method2_stage2",
                "_pid": int(pid),
                "_pair": jj + 1,
            })

    # ---- shuffle DPO data (stage 1 and 2 mixed) ----
    rng.shuffle(dpo_data)

    # ---- build SFT data ----
    for rec in sft_recs:
        sft_data.append(make_sft_record(rec))

    # ---- stats ----
    n_dpo = len(dpo_data)
    n_sft = len(sft_data)
    n_stage1 = sum(1 for d in dpo_data if d["_source"] == "method2_stage1")
    n_stage2 = sum(1 for d in dpo_data if d["_source"] == "method2_stage2")

    print(f"[stats] DPO pairs:  {n_dpo} (expected {k*2*j})", file=sys.stderr)
    print(f"[stats]   stage1:   {n_stage1}", file=sys.stderr)
    print(f"[stats]   stage2:   {n_stage2}", file=sys.stderr)
    print(f"[stats] SFT pairs:  {n_sft} (expected {sft_count})", file=sys.stderr)

    return dpo_data, sft_data


def main():
    ap = argparse.ArgumentParser(description="Build method2 joint DPO training data")
    ap.add_argument("--eval_result", required=True,
                    help="Path to eval_1_*_done.xlsx (S1 attack results with verdict column)")
    ap.add_argument("--rejections", required=True,
                    help="Path to collected_prompts_2000_rejections.xlsx")
    ap.add_argument("--sft_source", default="OCI", choices=["magicoder", "OCI"],
                    help="SFT data source: magicoder or OCI (default: OCI)")
    ap.add_argument("--sft_data", required=True,
                    help="Path to SFT source data file (magicoder_normalized.jsonl or OCI JSON)")
    ap.add_argument("--dpo_source", default=None, choices=["magicoder", "OCI"],
                    help="DPO random code source (default: same as --sft_source)")
    ap.add_argument("--dpo_data", default=None,
                    help="Path to DPO random code data file (default: same as --sft_data)")
    ap.add_argument("--output_dpo", required=True,
                    help="Output DPO .json path (LLaMA-Factory ShareGPT format)")
    ap.add_argument("--output_sft", required=True,
                    help="Output SFT .json path (LLaMA-Factory ShareGPT format)")
    ap.add_argument("--k", type=int, default=None,
                    help="Number of BAD prompts to use. DPO=2*j*k, SFT=2*j*k*SFT")
    ap.add_argument("--j", type=int, default=10,
                    help="Number of DPO pairs per prompt per stage (default 10). Total DPO = 2*j*k")
    ap.add_argument("--SFT", type=float, default=2.5,
                    help="SFT:DPO ratio (default 2.5). SFT count = 2*j*k * SFT")
    ap.add_argument("--sft_python_only", action="store_true",
                    help="Filter SFT records to Python only")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dpo_data, sft_data = build_method2_data(
        eval_result_path=args.eval_result,
        rejections_path=args.rejections,
        sft_data_path=args.sft_data,
        sft_source=args.sft_source,
        k=args.k,
        seed=args.seed,
        j=args.j,
        SFT=args.SFT,
        sft_python_only=args.sft_python_only,
        dpo_data_path=args.dpo_data,
        dpo_source=args.dpo_source,
    )

    # ---- write DPO ----
    out_dpo = Path(args.output_dpo)
    out_dpo.parent.mkdir(parents=True, exist_ok=True)
    with open(out_dpo, "w", encoding="utf-8") as f:
        json.dump(dpo_data, f, ensure_ascii=False, indent=2)
    print(f"[done] {len(dpo_data)} DPO samples -> {out_dpo}", file=sys.stderr)

    # ---- write SFT ----
    out_sft = Path(args.output_sft)
    out_sft.parent.mkdir(parents=True, exist_ok=True)
    with open(out_sft, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)
    print(f"[done] {len(sft_data)} SFT samples -> {out_sft}", file=sys.stderr)


if __name__ == "__main__":
    main()
