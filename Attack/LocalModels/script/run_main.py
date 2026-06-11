from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import llguidance
import llguidance.hf
from llguidance import LLMatcher
from llguidance.torch import (
    allocate_token_bitmask,
    fill_next_token_bitmask,
    apply_token_bitmask_inplace,
)

import pandas as pd
import time
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from datetime import datetime
import json
import re
import os
import argparse
import gc
import traceback
import signal


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
GRAMMAR_DIR = os.path.abspath(os.path.join(REPO_ROOT, "grammars"))

_SAMPLING = {
    "torch_dtype": torch.bfloat16,
    "temperature": 0.9,
    "top_k": None,
    "top_p": 0.95,
    "repetition_penalty": 1.2,
    "max_new_tokens": 1024,
}


def _cfg(name, prompt_filename=None, prompt_subdir="test"):
    """One-liner: filename = prompt_<name>.xlsx (or override), output = Results/<name>."""
    fname = prompt_filename if prompt_filename else f"prompt_{name}.xlsx"
    return {
        "prompt_file": os.path.join(REPO_ROOT, "../", "prompt", prompt_subdir, fname),
        "output_root": os.path.join(REPO_ROOT, "Results", name),
        **_SAMPLING,
    }


BENCHMARK_CONFIGS = {
    "rmc":       _cfg("rmc"),
    "mal":       _cfg("mal"),
    "trainset":  _cfg("trainset", prompt_filename="collected_prompts_2000.xlsx", prompt_subdir="train"),
}


LANGUAGE_ALIASES = {
    "python": "python",
    "py": "python",
    "c": "c",
    "c99": "c",
    "c11": "c",
    "c++": "cpp",
    "cpp": "cpp",
    "cxx": "cpp",
    "cc": "cpp",
    "java": "java",
    "golang": "go",
    "go": "go",
    "javascript": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "ts": "typescript",
    "c#": "csharp",
    "csharp": "csharp",
    "cs": "csharp",
    "rust": "rust",
    "php": "php",
}

LANGUAGE_REGEX_RULES = [
    ("cpp", re.compile(r"\bc\+\+\b|\bcpp\b|\bc\s*plus\s*plus\b", re.IGNORECASE)),
    ("csharp", re.compile(r"\bc#\b|\bcsharp\b", re.IGNORECASE)),
    ("python", re.compile(r"\bpython\b|\bpy\b", re.IGNORECASE)),
    ("java", re.compile(r"\bjava\b", re.IGNORECASE)),
    ("javascript", re.compile(r"\bjavascript\b|\bjs\b", re.IGNORECASE)),
    ("typescript", re.compile(r"\btypescript\b|\bts\b", re.IGNORECASE)),
    ("go", re.compile(r"\bgolang\b|\bgo\b", re.IGNORECASE)),
    ("rust", re.compile(r"\brust\b", re.IGNORECASE)),
    ("php", re.compile(r"\bphp\b", re.IGNORECASE)),
    ("c", re.compile(r"\bc\b", re.IGNORECASE)),
]

LANGUAGE_CODE_BLOCK = {
    "python": "python",
    "java": "java",
    "javascript": "javascript",
    "typescript": "typescript",
    "c": "c",
    "cpp": "cpp",
    "go": "go",
    "csharp": "csharp",
    "rust": "rust",
    "php": "php",
    "unknown": "python",
}

LANGUAGE_GRAMMAR_FILES = {
    "python": "python.lark",
    "java": "java.lark",
    "c": "c.lark",
    "cpp": "cpp.lark",
    "go": "go.lark",
}

DEFAULT_LANGUAGE = "python"
CODELANG_TO_LANGUAGE = {
    "py": "python",
    "cpp": "cpp",
    "java": "java",
}


model_info_dict = [
    {
        "mid": 0,
        "dir_name": "Meta-Llama-3-8B-Instruct",
        "local_path": "../models/Meta-Llama-3-8B-Instruct",
        "LLM_name": "Llama",
    },
    {
        "mid": 1,
        "dir_name": "Qwen2.5-7B-Instruct",
        "local_path": "../models/Qwen2.5-7B-Instruct",
        "LLM_name": "Qwen",
    },
    {
        "mid": 2,
        "dir_name": "Qwen2.5-Coder-7B-Instruct",
        "local_path": "../models/Qwen2.5-Coder-7B-Instruct",
        "LLM_name": "Qwen",
    }
]

model_name = "" 
model = ""
ll_tokenizer = ""
device = ""
tokenizer = ""
strategy_id = 1
benchmark_name = "rmc"
active_sampling = BENCHMARK_CONFIGS["rmc"]
grammar_cache = {}
show_generation_text = True
run_codelang = "py"
shard_id = 0        # 0-indexed shard index (0 = no sharding / first shard)
num_shards = 1      # total number of shards (1 = no sharding)
STRATEGY_RESULT_PREFIX = {
    0: "res_0",
    1: "res_1",
}

INTERRUPT_STATE = {
    "signal": None,
    "received_at": None,
}

def _handle_termination_signal(signum, frame):
    _ = frame
    try:
        sig_name = signal.Signals(signum).name
    except Exception:
        sig_name = f"SIG{signum}"

    INTERRUPT_STATE["signal"] = sig_name
    INTERRUPT_STATE["received_at"] = datetime.now().isoformat()
    print(f"\n[Signal] Received {sig_name}; saving checkpoint and exiting...", flush=True)
    raise KeyboardInterrupt(f"Received signal {sig_name}")


def register_signal_handlers():
    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
        try:
            signal.signal(sig, _handle_termination_signal)
        except Exception:
            # Some runtimes may not allow overriding specific signals.
            continue


def get_result_prefix(strategy):
    if strategy not in STRATEGY_RESULT_PREFIX:
        raise ValueError(f"Unsupported strategy for result naming: {strategy}")
    return STRATEGY_RESULT_PREFIX[strategy]


def build_result_file_name(strategy, model_name, benchmark_name, run_round, status, ext, codelang=None, timestamp=None):
    parts = [get_result_prefix(strategy)]
    if codelang:
        parts.append(codelang)
    if timestamp:
        parts.append(timestamp)
    parts.extend([model_name, benchmark_name])
    parts.extend([f"round{run_round}", status])
    return "_".join(parts) + f".{ext}"


def should_print_generation():
    return bool(show_generation_text)

def infer_resume_index(prompt_df):
    done_series = None
    if "done" in prompt_df.columns:
        done_series = parse_done_series(prompt_df["done"])

    response_completed = None
    if "response" in prompt_df.columns:
        response_series = prompt_df["response"].fillna("").astype(str)
        response_completed = response_series.str.strip() != ""

    if done_series is not None and response_completed is not None:
        # Compatibility path: some historical temp files have stale/invalid
        # done flags but valid response text. Treat either as completed.
        completed_mask = done_series | response_completed
    elif done_series is not None:
        completed_mask = done_series
    elif response_completed is not None:
        completed_mask = response_completed
    else:
        return 0

    pending_indices = completed_mask[~completed_mask]
    return int(pending_indices.index[0]) if len(pending_indices) > 0 else len(prompt_df)


def parse_done_series(done_series):
    """Normalize done column from mixed Excel dtypes into a strict boolean series."""
    if done_series is None:
        return None

    def _to_bool(value):
        if pd.isna(value):
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y", "t"}:
            return True
        if text in {"false", "0", "no", "n", "f", "", "none", "nan"}:
            return False
        return bool(text)

    return done_series.apply(_to_bool).astype(bool)


def load_checkpoint(checkpoint_file):
    if not os.path.exists(checkpoint_file):
        return None
    try:
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as e:
        print(f"[Warning] Failed to read checkpoint file {checkpoint_file}: {e}")
    return None


def save_checkpoint(checkpoint_file, payload):
    temp_checkpoint_file = checkpoint_file + ".tmp"
    with open(temp_checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(temp_checkpoint_file, checkpoint_file)


_ILLEGAL_XLSX_CHAR_RE = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")
MAX_EXCEL_CELL_CHARS = 32767
MAX_RESPONSE_CHARS = 32767


def sanitize_response_for_storage(response_text, max_chars=MAX_EXCEL_CELL_CHARS):
    """Remove XLSX-illegal control chars and clip overlong cells to keep runs resumable."""
    text = str(response_text)
    cleaned = _ILLEGAL_XLSX_CHAR_RE.sub("", text)
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    return cleaned


def sanitize_dataframe_for_excel(df):
    """Return a copy where all object/string cells are safe for openpyxl."""
    sanitized_df = df.copy()
    object_columns = sanitized_df.select_dtypes(include=["object", "string"]).columns
    for col in object_columns:
        sanitized_df[col] = sanitized_df[col].apply(
            lambda value: sanitize_response_for_storage(value) if isinstance(value, str) else value
        )
    return sanitized_df


def safe_to_excel(df, file_path, index=False):
    """Try normal write first; on illegal cell content, sanitize and retry once."""
    try:
        df.to_excel(file_path, index=index)
        return
    except Exception as e:
        print(f"[Warning] to_excel failed for {file_path}: {type(e).__name__}: {e}")
        print("[Info] Sanitizing string cells and retrying Excel write...")

    sanitized_df = sanitize_dataframe_for_excel(df)
    sanitized_df.to_excel(file_path, index=index)
    print(f"[Info] Excel write succeeded after sanitization: {file_path}")


def normalize_language(raw_language, fallback=DEFAULT_LANGUAGE):
    if raw_language is None:
        return fallback
    lang = str(raw_language).strip().lower()
    if not lang or lang == "nan":
        return fallback
    normalized = LANGUAGE_ALIASES.get(lang, lang)
    return normalized if normalized in LANGUAGE_CODE_BLOCK else fallback


def get_codelang_default_language():
    return CODELANG_TO_LANGUAGE.get(run_codelang, DEFAULT_LANGUAGE)


def extract_language_from_prompt(prompt_text, fallback=None):
    text = str(prompt_text)
    for language_name, pattern in LANGUAGE_REGEX_RULES:
        if pattern.search(text):
            return language_name
    if fallback is not None:
        return fallback
    return DEFAULT_LANGUAGE


def resolve_language(row):
    default_language = get_codelang_default_language()
    if "language" in row:
        # Keep prompt-based detection for empty/invalid language cells.
        lang = normalize_language(row["language"], fallback=None)
        if lang:
            return lang
    return extract_language_from_prompt(row.get("prompt", ""), fallback=default_language)


def enrich_prompt_languages(prompt_df):
    if "language_resolved" not in prompt_df.columns:
        prompt_df["language_resolved"] = prompt_df.apply(resolve_language, axis=1)
        return

    # Existing column may be float64 (e.g., initialized with NaN only).
    # Cast to object before writing string values via masked assignment.
    prompt_df["language_resolved"] = prompt_df["language_resolved"].astype(object)

    missing_mask = prompt_df["language_resolved"].isna() | (prompt_df["language_resolved"].astype(str).str.strip() == "")
    if missing_mask.any():
        prompt_df.loc[missing_mask, "language_resolved"] = prompt_df.loc[missing_mask].apply(resolve_language, axis=1)
    prompt_df["language_resolved"] = prompt_df["language_resolved"].apply(normalize_language)


def get_code_block_language(language_name):
    return LANGUAGE_CODE_BLOCK.get(language_name, "text")


def apply_strategy_to_prompt(prompt_text, language_name, strategy):
    """Wrap the raw prompt with any strategy-specific text.

    S0/S1: no wrapping — the attack for S1 is purely the forced decoding prefix
    handled separately in get_constrained_prefix().
    """
    _ = language_name
    return str(prompt_text)


def get_constrained_prefix(language_name, strategy):
    """Return (wrapper_prefix, grammar_prefix) for grammar-constrained strategies.

    The two parts are concatenated and injected at the start of the assistant
    turn, but they are treated differently by the matcher:

      wrapper_prefix : prepended to the context but NOT fed to the matcher,
                       because the markdown code fence (```python) is not valid
                       Python and python.lark cannot parse it.
      grammar_prefix : prepended AND fed to the matcher (consume_token), so the
                       grammar state advances to "in the middle of the import
                       statements" before constrained generation continues.

    Only S1 uses constrained decoding.
    """
    code_lang = get_code_block_language(language_name)
    if strategy == 1:
        return f"```{code_lang}\n", "import os\nimport "
    return "", ""


def load_lark_grammar(language_name):
    canonical_lang = normalize_language(language_name)
    grammar_lang = canonical_lang if canonical_lang in LANGUAGE_GRAMMAR_FILES else "python"
    if grammar_lang in grammar_cache:
        return grammar_cache[grammar_lang]

    grammar_file = os.path.join(GRAMMAR_DIR, LANGUAGE_GRAMMAR_FILES[grammar_lang])
    with open(grammar_file, "r", encoding="utf-8") as f:
        grammar_text = f.read()
    grammar_cache[grammar_lang] = grammar_text
    return grammar_text


def build_chat_template_prompt(user_message):
    """Render a single-turn user prompt via the tokenizer's chat_template.

    No explicit system turn is injected — each model's official template
    determines its own default system behavior (e.g., Qwen2.5 auto-adds
    "You are Qwen ..."; Llama-3 defaults to no system turn). This preserves
    the model's native safety alignment.

    add_generation_prompt=True ensures the template appends the assistant
    start token.
    """
    global tokenizer
    messages = [{"role": "user", "content": user_message}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def format_prompt_with_chat_template(user_message):
    """Use the tokenizer's built-in chat_template when available.

    Falls back to manual prompt formatting in format_prompt() if the
    checkpoint lacks a chat_template. New models require no registration
    — availability is determined entirely by the checkpoint.
    """
    chat_template = getattr(tokenizer, "chat_template", None)
    if not chat_template:
        return None
    try:
        return build_chat_template_prompt(user_message)
    except Exception as e:
        print(f"[Warning] apply_chat_template failed for {model_name}: {e}. Fallback to manual prompt format.")
        return None


# Format according to different LLM prompt chat templates
def format_prompt(user_message):
    global model_name
    chat_template_prompt = format_prompt_with_chat_template(user_message)
    if chat_template_prompt is not None:
        return chat_template_prompt

    if model_name in ["deepseek-coder-7b-instruct-v1.5", "deepseek-coder-6.7b-instruct", "deepseek-coder-33b-instruct"]:
        new_prompt = f"""You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
{ user_message }
### Response:
"""
        return  new_prompt
        
    return user_message


def get_response_unconstrained(prompt_items, batch_start_idx, total_rows, terminators):
    global model, tokenizer, active_sampling

    formatted_prompts = []
    for i, item in enumerate(prompt_items):
        global_idx = batch_start_idx + i + 1
        original_prompt = item["prompt"]
        language_name = normalize_language(item.get("language", DEFAULT_LANGUAGE))
        prompt_with_strategy = apply_strategy_to_prompt(original_prompt, language_name, strategy_id)
        formatted = format_prompt(prompt_with_strategy)

        formatted_prompts.append(formatted)
        if should_print_generation():
            print(f"\n[{global_idx}/{total_rows}] Start generating response...")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_width = inputs["input_ids"].shape[1]

    eos_token_id = terminators if len(terminators) > 1 else terminators[0]
    top_k_value = active_sampling["top_k"]
    if top_k_value is None:
        top_k_value = tokenizer.vocab_size

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=True,
            top_k=top_k_value,
            top_p=active_sampling["top_p"],
            temperature=active_sampling["temperature"],
            repetition_penalty=active_sampling["repetition_penalty"],
            num_return_sequences=1,
            eos_token_id=eos_token_id,
            max_new_tokens=active_sampling["max_new_tokens"],
            pad_token_id=tokenizer.eos_token_id,
        )

    responses = []
    for i in range(len(prompt_items)):
        global_idx = batch_start_idx + i + 1

        # Because we encode the prompt + prefix together using padding=True,
        # huggingface generate will append the new tokens directly after input_width.
        response_start = input_width

        if response_start < 0 or response_start > generated[i].shape[0]:
            response_start = input_width

        response_ids = generated[i][response_start:]
        only_response = tokenizer.decode(response_ids, skip_special_tokens=True)

        if should_print_generation():
            preview = only_response.strip().replace("\n", "\\n")
            if len(preview) > 160:
                preview = preview[:157] + "..."
            print(f"[{global_idx}/{total_rows}] Response preview: {preview or '<empty>'}")
        responses.append(only_response)

    return responses

    

def get_response(prompt_items, batch_start_idx, total_rows):
    global model, tokenizer, model_name, ll_tokenizer, device, active_sampling, strategy_id

    terminators = [tokenizer.eos_token_id]
    if model_name in ["Meta-Llama-3-8B-Instruct", "Meta-Llama-3-70B-Instruct"]:
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    elif model_name in [
        "Qwen2.5-Coder-7B-Instruct",
        "Qwen2.5-7B-Instruct"
    ]:
        qwen_im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(qwen_im_end_id, int) and qwen_im_end_id >= 0:
            terminators.append(qwen_im_end_id)

    # S0 (Vanilla): native generation with chat template, no attack.
    if strategy_id == 0:
        return get_response_unconstrained(prompt_items, batch_start_idx, total_rows, terminators)
        
    batch_size = len(prompt_items)
    responses = [""] * batch_size
    valid_mask = [True] * batch_size
    matchers = []
    
    all_input_ids = []
    generated_tokens_list = [[] for _ in range(batch_size)]
    generated_token_set_list = [set() for _ in range(batch_size)]
    
    effective_max_new_tokens_list = []
    stop_on_fence_list = []
    stop_fence_min_len_list = []
    
    for i, item in enumerate(prompt_items):
        global_idx = batch_start_idx + i + 1
        original_prompt = item["prompt"]
        language_name = normalize_language(item.get("language", "unknown"))
        prompt_with_strategy = apply_strategy_to_prompt(original_prompt, language_name, strategy_id)
        formatted = format_prompt(prompt_with_strategy) # Apply chat template with prompt
        
        grammar = LLMatcher.grammar_from_lark(load_lark_grammar(language_name))
        matcher = LLMatcher(ll_tokenizer, grammar, log_level=0)
        if matcher.is_error():
            print(f"Grammar error ({language_name}): {matcher.get_error()}")
            responses[i] = "grammar error"
            valid_mask[i] = False
            matchers.append(None)
            all_input_ids.append(torch.tensor([], dtype=torch.long, device=device))
            effective_max_new_tokens_list.append(0)
            stop_on_fence_list.append(False)
            stop_fence_min_len_list.append(0)
            continue
            
        matchers.append(matcher)
        
        input_ids = tokenizer(formatted, return_tensors="pt").input_ids[0].to(device)

        # wrapper_prefix (markdown fence) is injected as plain context only.
        # grammar_prefix is injected AND consumed by the matcher so the grammar
        # state lands mid-import before generation continues.
        # Encoding the two separately keeps the matcher token stream aligned
        # with the grammar input.
        wrapper_prefix, grammar_prefix = get_constrained_prefix(language_name, strategy_id)
        wrapper_tokens = tokenizer.encode(wrapper_prefix, add_special_tokens=False) if wrapper_prefix else []
        grammar_tokens = tokenizer.encode(grammar_prefix, add_special_tokens=False) if grammar_prefix else []

        for tok in grammar_tokens:
            if not matcher.consume_token(tok):
                print(f"\n  [Warn] matcher rejected forced prefix token {tok} for item {i}: {matcher.get_error()}")
                break

        prefix_tokens = wrapper_tokens + grammar_tokens
        if prefix_tokens:
            generated_tokens_list[i].extend(prefix_tokens)
            generated_token_set_list[i].update(prefix_tokens)
            input_ids = torch.cat([input_ids, torch.tensor(prefix_tokens, device=device)])
        all_input_ids.append(input_ids)
        
        effective_max_new_tokens_list.append(active_sampling["max_new_tokens"])
        # S1 stops early when the closing ``` fence appears
        stop_on_fence_list.append(True)
        stop_fence_min_len_list.append(len(generated_tokens_list[i]) + 4)
            
        if should_print_generation() and i == 0:
            print(f"\n[{global_idx}/{total_rows}] Batched start generating responses...")
            
    if not any(valid_mask):
        return responses
    # Left-pad all inputs to the same length
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    max_len = max(ids.shape[0] for i, ids in enumerate(all_input_ids) if valid_mask[i])
    
    batched_input_ids = []
    batched_attention_mask = []
    for i in range(batch_size):
        if not valid_mask[i]:
            batched_input_ids.append(torch.full((max_len,), pad_id, dtype=torch.long, device=device))
            batched_attention_mask.append(torch.zeros(max_len, dtype=torch.long, device=device))
        else:
            ids = all_input_ids[i]
            pad_len = max_len - ids.shape[0]
            padded_ids = torch.cat([torch.full((pad_len,), pad_id, dtype=torch.long, device=device), ids])
            attn_mask = torch.cat([torch.zeros(pad_len, dtype=torch.long, device=device), torch.ones(ids.shape[0], dtype=torch.long, device=device)])
            batched_input_ids.append(padded_ids)
            batched_attention_mask.append(attn_mask)
            
    batched_input_ids = torch.stack(batched_input_ids)
    batched_attention_mask = torch.stack(batched_attention_mask)
    
    past_key_values = None
    is_stopped = [not mask for mask in valid_mask]
    max_steps = max(effective_max_new_tokens_list) if effective_max_new_tokens_list else 0
    
    bitmask = allocate_token_bitmask(batch_size, ll_tokenizer.vocab_size)
    current_input_ids = batched_input_ids
    
    with torch.inference_mode():
        for step in range(max_steps):
            if all(is_stopped):
                break
                
            outputs = model(
                input_ids=current_input_ids,
                attention_mask=batched_attention_mask if past_key_values is None else None,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            
            next_tokens = [pad_id] * batch_size
            
            # 1. Fill bitmask and ff_tokens
            for i in range(batch_size):
                if is_stopped[i] or step >= effective_max_new_tokens_list[i]:
                    is_stopped[i] = True
                    continue
                    
                matcher = matchers[i]
                ff_tokens = matcher.compute_ff_tokens()
                if ff_tokens: # Only legal token
                    next_token = ff_tokens[0]
                    matcher.consume_token(next_token)
                    next_tokens[i] = next_token
                else:
                    # fill bitmask
                    fill_next_token_bitmask(matcher, bitmask[i:i+1])
                    
            # Apply bitmask (applies across batch dimension)
            apply_token_bitmask_inplace(logits, bitmask.to(device))
            
            # Sampling
            temperature = active_sampling.get("temperature", 1.0)
            if float(temperature) == 0.0:
                sampled = torch.argmax(logits, dim=-1)
            else:
                top_p = active_sampling.get("top_p", 1.0)
                top_k = active_sampling.get("top_k", None)
                if top_k is None or top_k <= 0:
                    top_k = logits.size(-1)
                
                for i in range(batch_size):
                    if is_stopped[i] or next_tokens[i] != pad_id: continue
                    # rep penalty
                    rep_pen = active_sampling.get("repetition_penalty", 1.0)
                    if rep_pen != 1.0:
                        for token_id in generated_token_set_list[i]:
                            if logits[i, token_id] > 0:
                                logits[i, token_id] /= rep_pen
                            else:
                                logits[i, token_id] *= rep_pen
                                
                    # filter top_k
                    values, indices = torch.topk(logits[i], min(top_k, logits.size(-1)))
                    logits[i, :] = float('-inf')
                    logits[i, indices] = values
                    
                    # filter top_p
                    sorted_logits, sorted_indices = torch.sort(logits[i], descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits / temperature, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[i, indices_to_remove] = float('-inf')

                probs = torch.softmax(logits / temperature, dim=-1)
                sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            # Process sampled
            for i in range(batch_size):
                if is_stopped[i]:
                    continue
                if next_tokens[i] == pad_id: # need to pick sampled token
                    next_token = sampled[i].item()
                    ok = matchers[i].consume_token(next_token)
                    if not ok:
                        print(f"\n  [Warn] matcher rejected token for item {i}: {matchers[i].get_error()}")
                        is_stopped[i] = True
                        continue
                    next_tokens[i] = next_token
                    
                cur_token = next_tokens[i]
                generated_tokens_list[i].append(cur_token)
                generated_token_set_list[i].add(cur_token)
                if should_print_generation() and i == 0:
                    print(tokenizer.decode([cur_token]), end="", flush=True)
                
                if matchers[i].is_stopped() or cur_token in terminators:
                    is_stopped[i] = True
                    
                # Check for early stopping at closing code fence
                if stop_on_fence_list[i] and len(generated_tokens_list[i]) >= stop_fence_min_len_list[i]:
                    tail = tokenizer.decode(generated_tokens_list[i][-8:], skip_special_tokens=True)
                    if "\n```" in tail or tail.rstrip().endswith("```"):
                        is_stopped[i] = True
                        
            current_input_ids = torch.tensor(next_tokens, dtype=torch.long, device=device).unsqueeze(1)
            batched_attention_mask = torch.cat([batched_attention_mask, torch.ones((batch_size, 1), dtype=torch.long, device=device)], dim=-1)

    for i in range(batch_size):
        if not valid_mask[i]:
            continue
        only_response = tokenizer.decode(generated_tokens_list[i], skip_special_tokens=True)
        responses[i] = only_response
        if should_print_generation():
            preview = only_response.strip().replace("\n", "\\n")
            if len(preview) > 160: preview = preview[:157] + "..."
            print(f"\n[{batch_start_idx + i + 1}/{total_rows}] Batched Response preview: {preview or '<empty>'}")
            
    return responses


def change_LLM_name_in_prompt(old_prompt,LLM_name):
    if LLM_name == "chatgpt" :
        return old_prompt
    
    result_str = re.sub('chatgpt', LLM_name, old_prompt, flags=re.IGNORECASE)
    return result_str


def resolve_prompt_file(run_round):
    _ = run_round
    if run_codelang == "py":
        prompt_file = BENCHMARK_CONFIGS[benchmark_name]["prompt_file"]
    else:
        prompt_file = os.path.join(REPO_ROOT, "prompt", "test", "codelang", f"prompt_{benchmark_name}_{run_codelang}.xlsx")

    if os.path.exists(prompt_file):
        return prompt_file

    raise FileNotFoundError(
        f"Prompt file not found: {prompt_file}. "
        "Please generate language-specific xlsx files first (for example: prompt_mal_cpp.xlsx)."
    )


def resolve_model_path(model_info):
    candidates = []
    if "local_path" in model_info and model_info["local_path"]:
        candidates.append(model_info["local_path"])
    candidates.append(os.path.join(REPO_ROOT, "models", model_info["dir_name"]))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"Model path not found for {model_info['dir_name']}. Tried local paths: {candidates}"
    )


def release_model_resources(release_tokenizer=False):
    global model, tokenizer, ll_tokenizer

    model = None
    if release_tokenizer:
        tokenizer = None
        ll_tokenizer = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _shard_suffix():
    """Return a filename suffix like '_shard2of4' when sharding is active."""
    if num_shards <= 1:
        return ""
    return f"_shard{shard_id}of{num_shards}"


def run_single_benchmark(file_path, output_file_dir, run_round, llm_name):
    global model_name, benchmark_name, active_sampling, strategy_id, run_codelang

    start_time = time.perf_counter()

    _sfx = _shard_suffix()
    temp_file = os.path.join(
        output_file_dir,
        build_result_file_name(
            strategy_id,
            model_name,
            benchmark_name,
            run_round,
            f"temp{_sfx}",
            "xlsx",
            codelang=run_codelang,
        ),
    )
    checkpoint_file = temp_file + ".ckpt.json"

    if os.path.exists(temp_file):
        print(f"Found temp file {temp_file}, resuming from breakpoint...")
        prompt_df = pd.read_excel(temp_file)
        if 'response' not in prompt_df.columns:
            prompt_df['response'] = ''
        prompt_df['response'] = prompt_df['response'].fillna('')

        if 'done' in prompt_df.columns:
            prompt_df['done'] = parse_done_series(prompt_df['done'])
        else:
            prompt_df['done'] = False

        inferred_index = infer_resume_index(prompt_df)
        checkpoint_data = load_checkpoint(checkpoint_file)
        if checkpoint_data and checkpoint_data.get("interrupted"):
            prev_type = checkpoint_data.get("error_type") or "Interrupted"
            prev_error = checkpoint_data.get("error") or "(no message)"
            prev_signal = checkpoint_data.get("signal")
            if prev_signal:
                print(f"[Info] Previous run was interrupted by {prev_signal}: {prev_type}: {prev_error}")
            else:
                print(f"[Info] Previous run interruption detected: {prev_type}: {prev_error}")

        if checkpoint_data and isinstance(checkpoint_data.get("next_index"), int):
            checkpoint_index = int(checkpoint_data["next_index"])
            if checkpoint_index < 0 or checkpoint_index > len(prompt_df):
                print(f"[Warning] Invalid checkpoint index {checkpoint_index}; fallback to inferred index {inferred_index}.")
                index = inferred_index
            else:
                # Prefer the furthest recoverable position if sidecar and
                # table disagree; sidecar may lag after abrupt interruption.
                index = max(checkpoint_index, inferred_index)
                if checkpoint_index != inferred_index:
                    print(
                        "[Warning] Checkpoint mismatch detected "
                        f"(sidecar={checkpoint_index}, inferred={inferred_index}), use {index}."
                    )
        else:
            index = inferred_index

        print(f"Resuming from index {index}")
    else:
        prompt_df = pd.read_excel(file_path)

        # ── Shard slicing ────────────────────────────────────────────────────
        if num_shards > 1:
            total_full = len(prompt_df)
            rows_per_shard = (total_full + num_shards - 1) // num_shards
            row_start = shard_id * rows_per_shard
            row_end   = min(row_start + rows_per_shard, total_full)
            prompt_df = prompt_df.iloc[row_start:row_end].reset_index(drop=True)
            print(f"[Shard {shard_id}/{num_shards}] Using rows {row_start}-{row_end-1} "
                  f"({len(prompt_df)} / {total_full} rows)")
        # ─────────────────────────────────────────────────────────────────────

        prompt_df['response'] = ''
        prompt_df['done'] = False
        index = 0

    enrich_prompt_languages(prompt_df)

    # samples num
    num_rows = len(prompt_df)
    print("total row:", num_rows)

    batch_size = 8
    error_id_list = []
    try:
        while index < num_rows:
            batch_end = min(index + batch_size, num_rows)
            batch_df = prompt_df.iloc[index:batch_end]

            batch_items = []
            for i, row in batch_df.iterrows():
                prompt = row['prompt']
                # 'category' and 'level' only exist in rmc/mal benchmarks.
                # pku and other benchmarks without these columns skip this branch.
                if row.get('category') == "text-to-code" and row.get("level") == 3:
                    prompt = change_LLM_name_in_prompt(prompt, llm_name)
                    prompt_df.at[i, 'prompt'] = prompt
                language_name = row.get("language_resolved", "unknown")
                batch_items.append({"prompt": prompt, "language": language_name})

            same_error_times=0
            while same_error_times < 3:
                try:
                    responses = get_response(batch_items, index, num_rows)
                    break
                except Exception as e:
                    err_text = str(e)
                    # CUDA fatal errors (device-side assert, illegal memory access)
                    # poison the entire CUDA context. All subsequent kernel launches
                    # will fail immediately. Retrying is pointless — it would fill
                    # every remaining batch with "Error: Failed after 3 retries."
                    # Instead, raise and let the outer except BaseException save a
                    # checkpoint so the run can resume from this batch.
                    fatal_cuda_markers = (
                        "CUDA error: device-side assert",
                        "CUDA error: an illegal memory access",
                        "CUBLAS_STATUS_EXECUTION_FAILED",
                        "CUDNN_STATUS_EXECUTION_FAILED",
                    )
                    if any(marker in err_text for marker in fatal_cuda_markers):
                        print(f"\n[Fatal] CUDA context poisoned at batch {index}-{batch_end}:\n  {err_text}")
                        raise
                    same_error_times+=1
                    import traceback as _tb
                    print(f"\n[Warning] Error at batch {index}-{batch_end}: {repr(e)} | Retrying ({same_error_times}/3)...")
                    _tb.print_exc()
                    if same_error_times >= 3:
                        print(f'\nBatch {index}-{batch_end} failed completely.')
                        error_id_list.extend(range(index, batch_end))
                        responses = ["Error: Failed after 3 retries."] * len(batch_items)

            for offset, response in enumerate(responses):
                idx = index + offset
                response_text = str(response)
                if len(response_text) > MAX_RESPONSE_CHARS:
                    print(
                        f"[Warning] response too long at row {idx} "
                        f"({len(response_text)} chars); truncating to {MAX_RESPONSE_CHARS}."
                    )
                    response_text = response_text[:MAX_RESPONSE_CHARS]
                response_text = sanitize_response_for_storage(response_text, max_chars=MAX_RESPONSE_CHARS)
                prompt_df.at[idx, 'response'] = response_text
                prompt_df.at[idx, 'done'] = True

            # Save progress to temp file for breakpoint resume.
            safe_to_excel(prompt_df, temp_file, index=False)

            index = batch_end

            save_checkpoint(checkpoint_file, {
                "next_index": int(index),
                "num_rows": int(num_rows),
                "timestamp": datetime.now().isoformat(),
                "model_name": model_name,
                "benchmark": benchmark_name,
                "strategy": strategy_id,
                "run_round": run_round,
            })

            # Compute ETA and progress
            current_time = time.perf_counter()
            elapsed_time = current_time - start_time
            avg_time = elapsed_time / max(index, 1)
            remain_items = num_rows - index
            eta_secs = int(avg_time * remain_items)

            h, rem = divmod(eta_secs, 3600)
            m, s = divmod(rem, 60)
            eta_str = f"{h:02d}h {m:02d}m {s:02d}s" if h > 0 else f"{m:02d}m {s:02d}s"

            print(f"\r[Round {run_round}][{benchmark_name}] Progress: {index}/{num_rows} ({(index)/num_rows*100:.1f}%) | Elapsed: {int(elapsed_time)}s | ETA: {eta_str}", end="", flush=True)
    except BaseException as e:
        # Persist state before re-raising so interrupted runs can always resume.
        error_type = type(e).__name__
        error_text = str(e).strip() or repr(e)
        signal_name = INTERRUPT_STATE.get("signal")
        if signal_name:
            error_text = f"{error_text} (signal={signal_name})"
        print(f"\n[Error] Run interrupted at index {index}: {error_type}: {error_text}")
        try:
            safe_to_excel(prompt_df, temp_file, index=False)
            save_checkpoint(checkpoint_file, {
                "next_index": int(index),
                "num_rows": int(num_rows),
                "timestamp": datetime.now().isoformat(),
                "interrupted": True,
                "error_type": error_type,
                "error": error_text,
                "signal": signal_name,
                "signal_received_at": INTERRUPT_STATE.get("received_at"),
                "traceback": traceback.format_exc(),
                "model_name": model_name,
                "benchmark": benchmark_name,
                "strategy": strategy_id,
                "run_round": run_round,
            })
            print(f"[Info] Checkpoint saved after interruption: {checkpoint_file}")
        except Exception as save_error:
            print(f"[Error] Failed to persist checkpoint during interruption: {save_error}")
        raise

    print()
        
    timestamp = time.strftime("%Y%m%d%H%M%S")
    _sfx = _shard_suffix()

    output_file = os.path.join(
        output_file_dir,
        build_result_file_name(
            strategy_id,
            model_name,
            benchmark_name,
            run_round,
            f"done{_sfx}",
            "xlsx",
            codelang=run_codelang,
            timestamp=timestamp,
        ),
    )
    print("save to:",output_file)

    json_name = os.path.join(
        output_file_dir,
        build_result_file_name(
            strategy_id,
            model_name,
            benchmark_name,
            run_round,
            f"done{_sfx}",
            "json",
            codelang=run_codelang,
            timestamp=timestamp,
        ),
    )
    prompt_df.to_json(json_name,orient='records')
    safe_to_excel(prompt_df, output_file, index=False)

    
    end_time = time.perf_counter()
    e2e_inference_time = (end_time-start_time)
    print(f"The time consuming is {e2e_inference_time} s")

    
    run_info = {
        "model_name":model_name,
        "run_time":timestamp,
        "time consuming":f"{e2e_inference_time} s",
        "error_id_list":error_id_list,
        "benchmark": benchmark_name,
        "codelang": run_codelang,
        "strategy": strategy_id,
        "sampling": {
            "temperature": active_sampling["temperature"],
            "top_k": active_sampling["top_k"],
            "top_p": active_sampling["top_p"],
            "repetition_penalty": active_sampling["repetition_penalty"],
            "max_new_tokens": active_sampling["max_new_tokens"],
        },
        "show_generation_text": bool(show_generation_text),
    }
   
    run_info_path = os.path.join(
        output_file_dir,
        f"run_info_{model_name}_{benchmark_name}_{run_codelang}_s{strategy_id}_{timestamp}_round{run_round}_done.json",
    )
    with open(run_info_path, 'w') as f:
        json.dump(run_info, f)
        
    if os.path.exists(temp_file):
        os.remove(temp_file)
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    print("done! output save to:", output_file_dir)


def main():

    # logo
    print("""
 ██████╗ ██████╗ ██████╗ ███████╗███████╗██████╗ ███████╗ █████╗ ██████╗ 
██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔════╝██╔══██╗██╔════╝██╔══██╗██╔══██╗
██║     ██║   ██║██║  ██║█████╗  ███████╗██████╔╝█████╗  ███████║██████╔╝
██║     ██║   ██║██║  ██║██╔══╝  ╚════██║██╔═══╝ ██╔══╝  ██╔══██║██╔══██╗
╚██████╗╚██████╔╝██████╔╝███████╗███████║██║     ███████╗██║  ██║██║  ██║
 ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝
    """)
    
    global model_info_dict, model, ll_tokenizer, device, tokenizer, model_name
    global strategy_id, benchmark_name, active_sampling, show_generation_text, run_codelang
    global shard_id, num_shards

    register_signal_handlers()
    

    parser = argparse.ArgumentParser(description='parsers')
    parser.add_argument('--run_model_index',type=str,nargs='+',required=True, help='model_index (e.g. 10 11 12 or 10,11,12)')
    parser.add_argument('--run_rounds', type=int,nargs='+', default=[1], help='The number of rounds for this experiment (can be multiple rounds), with numbers separated by spaces')
    parser.add_argument('--strategy', type=int, nargs='+', choices=[0, 1], required=True, help='Decoding strategy ID(s): 0=Vanilla (native generation with chat template), 1=Ours (grammar-constrained decoding with forced import prefix)')
    parser.add_argument('--benchmark', type=str, nargs='+', choices=['rmc', 'mal', 'total', 'trainset'], required=True, help='Benchmark split(s) that control prompt file and hyperparameters. Use "total" to run both rmc and mal. Use "pku" for PKU-SafeRLHF DPO data collection. Use "trainset" for collected_prompts_2000.xlsx.')
    parser.add_argument('--codelang', type=str, choices=['py', 'cpp', 'java'], default='py', help='Code language selector. py uses base prompt file; cpp/java use prompt_{benchmark}_{codelang}.xlsx')
    parser.add_argument('--quiet_generation', action='store_true', help='Disable per-token/per-sample generation text output; only keep progress and ETA logs')
    parser.add_argument('--shard_id',   type=int, default=0, help='0-indexed shard index for data-parallel multi-GPU runs (default: 0)')
    parser.add_argument('--num_shards', type=int, default=1, help='Total number of shards; 1 = no sharding (default: 1)')

    args = parser.parse_args()


    run_model_indices_raw = args.run_model_index
    run_model_indices = []
    for rm in run_model_indices_raw:
        if ',' in rm:
            run_model_indices.extend([int(x) for x in rm.split(',')])
        else:
            run_model_indices.append(int(rm))

    run_rounds = args.run_rounds
    strategy_ids = list(dict.fromkeys(args.strategy))
    strategy_id = strategy_ids[0]
    
    benchmark_names = []
    for b in list(dict.fromkeys(args.benchmark)):
        if b == 'total':
            if 'rmc' not in benchmark_names:
                benchmark_names.append('rmc')
            if 'mal' not in benchmark_names:
                benchmark_names.append('mal')
        else:
            if b not in benchmark_names:
                benchmark_names.append(b)
                
    benchmark_name = benchmark_names[0]
    active_sampling = BENCHMARK_CONFIGS[benchmark_name]
    run_codelang = args.codelang
    show_generation_text = not args.quiet_generation
    shard_id   = args.shard_id
    num_shards = args.num_shards
    if num_shards > 1:
        print(f"[Shard mode] shard_id={shard_id}, num_shards={num_shards}")

    print("run_model_indices:", run_model_indices)
    print("run_rounds:", run_rounds)
    print("strategies:", strategy_ids)
    print("benchmarks:", benchmark_names)
    print("codelang:", run_codelang, "(default language for prompts without explicit language:", get_codelang_default_language() + ")")
    print("show_generation_text:", show_generation_text)
    for current_benchmark in benchmark_names:
        current_sampling = BENCHMARK_CONFIGS[current_benchmark]
        print(f"sampling[{current_benchmark}]:", {
            "temperature": current_sampling["temperature"],
            "top_k": current_sampling["top_k"],
            "top_p": current_sampling["top_p"],
            "repetition_penalty": current_sampling["repetition_penalty"],
            "max_new_tokens": current_sampling["max_new_tokens"],
        })
    
    import itertools
    for run_model_index, run_round, current_strategy in itertools.product(run_model_indices, run_rounds, strategy_ids):
        strategy_id = current_strategy
        print(f"Evaluating model_index: {run_model_index}, round: {run_round}, strategy: {strategy_id}")

        model_info = next((m for m in model_info_dict if m["mid"] == run_model_index), None)
        if model_info is None:
            print(f"Error: model with mid={run_model_index} not found in model_info_dict")
            continue
        model_path = resolve_model_path(model_info)
        LLM_name = model_info["LLM_name"] 

        print("******************************Key parameters for this run:******************************")
        print("model_path:",model_path)
        print("LLM_name:",LLM_name)
        print("benchmarks:", benchmark_names)
        print("****************************************************************************")

        model_name = model_path.split('/')[-1] 

        print("loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if tokenizer.pad_token_id is None:
            fallback_pad_token = tokenizer.eos_token or tokenizer.unk_token
            if fallback_pad_token is None:
                raise ValueError(f"Tokenizer for {model_name} has no eos_token/unk_token to reuse as pad_token.")
            tokenizer.pad_token = fallback_pad_token
        tokenizer.padding_side = "left"
        ll_tokenizer = llguidance.hf.from_tokenizer(tokenizer)

        load_dtype = torch.bfloat16

        print(f"loading model with dtype {load_dtype} for benchmarks: {benchmark_names}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            dtype=load_dtype,
            trust_remote_code=True,
            local_files_only=True,
        )
        model.eval()
        device = next(model.parameters()).device
        print("Loading completed!")
        print("🤩🤩🤩🤩🤩 "+model_name+",go!!!🤩🤩🤩🤩🤩")

        for current_benchmark in benchmark_names:
            benchmark_name = current_benchmark
            active_sampling = BENCHMARK_CONFIGS[benchmark_name]
            file_path = resolve_prompt_file(run_round)
            output_file_dir = os.path.join(active_sampling["output_root"], model_info["dir_name"], f"round{run_round}")

            print("******************************Current benchmark parameters:******************************")
            print("benchmark:", benchmark_name)
            print("prompt_file:", file_path)
            print("output_file_dir:", output_file_dir)
            print("sampling:", {
                "temperature": active_sampling["temperature"],
                "top_k": active_sampling["top_k"],
                "top_p": active_sampling["top_p"],
                "repetition_penalty": active_sampling["repetition_penalty"],
                "max_new_tokens": active_sampling["max_new_tokens"],
                "torch_dtype": str(active_sampling["torch_dtype"]),
            })
            print("****************************************************************************")

            os.makedirs(output_file_dir, exist_ok=True)

            run_single_benchmark(file_path, output_file_dir, run_round, LLM_name)

        release_model_resources(release_tokenizer=True)

    print("All rounds of all models have been executed!")

if __name__ == "__main__":
    main()