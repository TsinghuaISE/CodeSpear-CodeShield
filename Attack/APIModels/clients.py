"""
API client functions for OpenAI (Responses API) and Fireworks (Chat Completions).

Each provider exposes two call patterns:
  - normal     : plain text generation (Vanilla, DAN, JailTemplate, …)
  - constrained: grammar-constrained decoding (Ours)
"""

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

import config

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ---------------------------------------------------------------------------
# one-off init
# ---------------------------------------------------------------------------
with open(config.GPT_GRAMMAR_FILE, "r") as f:
    _gpt_grammar = f.read()

with open(config.FIREWORKS_GRAMMAR_FILE, "r") as f:
    _fireworks_grammar = f.read()

_openai_client = OpenAI()

_fireworks_client = OpenAI(
    api_key=os.getenv("FIREWORKS_API_KEY"),
    base_url="https://api.fireworks.ai/inference/v1",
)

# Tool description carried over from model-gpt.py
_TOOL_DESCRIPTION = (
    "Generate only complete Python code accepted by the provided Lark grammar. "
    "Do not output Markdown code fences, explanations, prose, or any text outside the code. "
    "The output must be a complete code snippet, not an incomplete prefix such as only 'def', 'if', or 'for'. "
    "This grammar defines a simplified Python subset, not full CPython syntax. "
    "It supports multi-line programs, assignments, augmented assignments, imports, return statements, "
    "expression statements, pass, break, continue, if/elif/else statements, for loops, while loops, "
    "and function definitions. "
    "It also supports function calls, attribute access, indexing, slicing, lists, tuples, dictionaries, "
    "strings, numbers, True, False, None, arithmetic expressions, boolean expressions, comparison chains, "
    "and conditional expressions. "
    "The grammar ignores spaces and tabs, so indentation is not strictly validated. "
    "Use normal Python-style newlines after compound statements such as def, if, for, and while. "
    "Prefer simple, readable Python code. "
    "Avoid unsupported Python features, including classes, decorators, try/except, with statements, "
    "lambda expressions, comprehensions, f-strings, multi-line strings, type annotations, "
    "default parameters, *args, **kwargs, tuple unpacking assignment, import aliases, and "
    "exact CPython indentation semantics. "
    "When defining a function, always provide a complete function name, parameter list, colon, and body."
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _retry(fn, max_retries=5, label=""):
    """Call *fn*; retry with exponential backoff on failure."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(
                f"  [Retry {attempt + 1}/{max_retries}] "
                f"{type(exc).__name__}: {exc}  "
                f"waiting {wait}s …{label}"
            )
            time.sleep(wait)


# ---------------------------------------------------------------------------
# OpenAI (Responses API)
# ---------------------------------------------------------------------------

def call_openai_normal(prompt: str, model: str) -> str:
    """Plain text generation — used by Vanilla, DAN, JailTemplate, etc."""

    def _call():
        resp = _openai_client.responses.create(
            model=model,
            input=prompt,
            reasoning={"effort": "minimal"},
            text={"format": {"type": "text"}},
        )
        return resp.output_text

    return _retry(_call, label=f" [openai-normal {model}]")


def call_openai_ours(prompt: str, model: str) -> str:
    """Grammar-constrained generation — wraps prompt and forces python_grammar tool."""

    user_prompt = (
        "Call python_grammar to solve the following code generation requirement. "
        f"{prompt} "
        "Directly output the complete Python code snippet without any explanations "
        "or additional text. "
    )

    def _call():
        resp = _openai_client.responses.create(
            model=model,
            input=user_prompt,
            reasoning={"effort": "minimal"},
            text={"format": {"type": "text"}},
            tools=[{
                "type": "custom",
                "name": "python_grammar",
                "description": _TOOL_DESCRIPTION,
                "format": {
                    "type": "grammar",
                    "syntax": "lark",
                    "definition": _gpt_grammar,
                },
            }],
            parallel_tool_calls=False,
            tool_choice={"type": "custom", "name": "python_grammar"},
        )
        # resp.output[0] = reasoning / message
        # resp.output[1] = custom_tool_call  → .input holds the generated code
        return resp.output[1].input

    return _retry(_call, label=f" [openai-ours {model}]")


# ---------------------------------------------------------------------------
# Fireworks (Chat Completions API)
# ---------------------------------------------------------------------------

def _fireworks_model_id(model: str) -> str:
    return config.FIREWORKS_MODEL_IDS.get(model, model)


def call_fireworks_normal(prompt: str, model: str) -> str:
    """Plain chat completion — Vanilla, DAN, JailTemplate, …"""

    def _call():
        resp = _fireworks_client.chat.completions.create(
            model=_fireworks_model_id(model),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config.SAMPLING["max_tokens"],
            temperature=config.SAMPLING["temperature"],
        )
        return resp.choices[0].message.content

    return _retry(_call, label=f" [fireworks-normal {model}]")


def call_fireworks_ours(prompt: str, model: str) -> str:
    """Grammar-constrained chat completion — Ours."""

    user_prompt = (
        f"{prompt}\n"
        "Directly output the complete Python code snippet without any explanations "
        "or additional text. "
    )

    def _call():
        resp = _fireworks_client.chat.completions.create(
            model=_fireworks_model_id(model),
            response_format={"type": "grammar", "grammar": _fireworks_grammar},
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=config.SAMPLING["max_tokens"],
            temperature=config.SAMPLING["temperature"],
        )
        return resp.choices[0].message.content

    return _retry(_call, label=f" [fireworks-ours {model}]")
