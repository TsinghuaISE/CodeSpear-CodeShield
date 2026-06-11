"""
Configuration for the API model evaluation system.

Models, benchmarks, methods, xlsx file paths, sampling parameters,
and output directory.
"""

import os

# ---- Paths ----
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
BENCH_ROOT = os.path.join(os.path.dirname(REPO_ROOT), "Bench")

# ---- Models ----
OPENAI_MODELS = ["gpt-5-mini", "gpt-5"]

FIREWORKS_MODELS = ["minimax", "gptoss", "minimax25", "gptossmini"]

# Fireworks model IDs (shorthand -> full API model ID)
FIREWORKS_MODEL_IDS = {
    "minimax": "accounts/fireworks/models/minimax-m2p7",
    "gptoss":  "accounts/fireworks/models/gpt-oss-120b",
    "minimax25": "accounts/fireworks/models/minimax-m2p5",
    "gptossmini": "accounts/fireworks/models/gpt-oss-20b",
}

# ---- Methods (use exact names as specified) ----
VANILLA             = "Vanilla"
OURS                = "Ours"

ALL_METHODS = [VANILLA, OURS]

# ---- Benchmarks ----
MAL = "mal"
RMC = "rmc"
ALL_BENCHMARKS = [MAL, RMC]

# ---- Prompt files ----
# Methods that read from the vanilla prompt file:
#   VANILLA, DAN, OURS
VANILLA_PROMPT_FILES = {
    MAL: os.path.join(BENCH_ROOT, "data", "xlsx", "prompt_mal.xlsx"),
    RMC: os.path.join(BENCH_ROOT, "data", "xlsx", "prompt_rmc.xlsx"),
}

# ---- Sampling parameters (consistent with run_main.py) ----
# Source: MalwareBench (Li et al., ACL 2025), Appendix A.3:
#   temperature=0.9, top_p=0.95.  max_tokens=1024 is project convention.
SAMPLING = {
    "temperature": 0.9,
    "top_p": 0.95,
    "max_tokens": 1024,
}

# ---- Output ----
OUTPUT_DIR = os.path.join(REPO_ROOT, "Results")

# ---- Grammar files ----
GPT_GRAMMAR_FILE       = os.path.join(REPO_ROOT, "grammar-gpt.lark")
FIREWORKS_GRAMMAR_FILE = os.path.join(REPO_ROOT, "grammar-fireworks.txt")

# ---- DAN model aliases ----
DAN_MODEL_ALIASES = {
    "gpt-5-mini": "ChatGPT",
    "gpt-5":      "ChatGPT",
    "minimax":    "assistant",
    "gptoss":     "assistant",
}
