# Grammar-Constrained Decoding Can Jailbreak LLMs into Generating Malicious Code

Official implementation of our paper "Grammar-Constrained Decoding Can Jailbreak LLMs into Generating Malicious Code".

<p align="center">
    <a href="https://www.python.org/">
        <img alt="Python" src="https://img.shields.io/badge/Python-3.11+-1f425f.svg?color=purple">
    </a>
    <a href="https://opensource.org/licenses/MIT">
        <img alt="License" src="https://img.shields.io/badge/License-MIT-blue">
    </a>
</p>

## 📰 News

* **[Jun, 2026]**: We've released the code for our paper "Grammar-Constrained Decoding Can Jailbreak LLMs into Generating Malicious Code".

## 👋 Overview

**CodeSpear** is a new jailbreak attack that leverages **grammar-constrained decoding (GCD)** to induce large language models (LLMs) into generating malicious code at a low cost. GCD was originally designed to improve the reliability of code generation by constraining LLMs to produce outputs that conform to a target grammar, but our paper shows that even benign, off-the-shelf code grammars can be weaponized to induce malicious code generation. In our threat model, malicious code generation requests may involve code intended for denial-of-service attacks, malware implementation, or credential theft.

This repository implements **CodeSpear** and evaluates it on two malicious code generation benchmarks (**RMCBench** and **MalwareBench**), revealing safety vulnerabilities in both **locally deployed LLMs** and **commercial API-based LLMs**.

![alt text](assets/codespear.png)

Our method achieves strong jailbreak performance across all tested models.

![alt text](assets/codespear_results.png)

To counter these threats, we also introduce **CodeShield**, a safety alignment approach for the **code modality** that trains the model to generate **honeypot code** against **CodeSpear** while preserving benign utility.

![alt text](assets/codeshield.png)

Our defense model maintains strong safety against the **CodeSpear attack** while **preserving general coding capability**.

![alt text](assets/codeshield_results.png)

The project includes:
- **Attack / LocalModels** — Run our jailbreak method against HuggingFace models with `llguidance`-based GCD
- **Attack / APIModels** — Evaluate commercial APIs (OpenAI, Fireworks) with grammar-constrained tool use
- **Defense** — DPO training data construction and safety alignment to defend against **CodeSpear**

## 📂 Directory Structure

```
CodeSpear-CodeShield/
├── Attack/
│   ├── APIModels/                # Commercial API-based LLMs evaluation
│   ├── LocalModels/              # Locally deployed LLMs evaluation
│   │   ├── grammars/             # Lark grammar files for GCD
│   │   │   ├── python.lark
│   │   │   ├── python_no_comments.lark
│   │   │   ├── java.lark
│   │   │   ├── c.lark
│   │   │   ├── cpp.lark
│   │   │   └── go.lark
│   │   ├── models/               # Local model checkpoints here
│   │   └── script/               # Run & evaluation scripts
│   │       ├── run_main.py       # Main entry point
│   │       ├── run_parallel.sh   # Multi-GPU parallel launcher with sharding
│   │       ├── run.sh            # Example single-GPU commands
│   │       ├── eval.sh           # Example evaluation commands
│   │       ├── evaluate_asr.py   # ASR evaluation (DeepSeek judge)
│   │       ├── evaluate_mr.py    # MR evaluation (DeepSeek judge)
│   │       └── merge_shards.py   # Merge sharded results
│   ├── prompt/                   # Prompt datasets (xlsx format)
│   │   ├── test/                 # Test prompts (RMCBench, MalwareBench, PKU-SafeRLHF)
│   │   │   └── codelang/         # Multi-language prompts (cpp, java, go, c)
│   │   └── train/                # Training prompts
│   └── .env.example              # API key template
├── Defense/
│   ├── build_training_data.py    # DPO training data construction
│   ├── train.py                  # CodeShield training script
│   ├── merge_config/             # llamafactory export configs
│   │   ├── merge_coder7b.yaml
│   │   ├── merge_7b.yaml
│   │   └── merge_llama3.yaml
│   └── data/                     # Training data of CodeShield
├── pyproject.toml
├── LICENSE
└── README.md
```

## 🚀 Environment Setup

Install dependencies via `uv` (recommended) or `pip`:

```bash
# Using uv (recommended)
uv sync

# Using pip
pip install -e.
```

### API Keys

Copy the environment template and set your API keys:

```bash
cp Attack/.env.example Attack/.env
# Edit Attack/.env with your actual API keys
```

Required environment variables:
- `OPENAI_API_KEY` — OpenAI API key
- `FIREWORKS_API_KEY` — Fireworks.AI API key
- `DEEPSEEK_API_KEY` — DeepSeek API key (for evaluation judges)

## 💽 Usage

### Attack: locally deployed LLMs

All commands are run from the `Attack/LocalModels/` directory.

```bash
cd Attack/LocalModels

# Run strategy 1 (grammar-constrained) on RMC benchmark (Python) with Qwen2.5-Coder-7B
CUDA_VISIBLE_DEVICES=0 python script/run_main.py \
    --run_model_index 2 --strategy 1 --codelang py --benchmark rmc

# Run baseline + GCD attack on all benchmarks with Llama-3-8B
CUDA_VISIBLE_DEVICES=0 python script/run_main.py \
    --run_model_index 0 --strategy 0 1 --codelang py --benchmark total
```

**Key arguments for `run_main.py`:**

| Argument | Description |
|----------|-------------|
| `--run_model_index` | Model index (see model index table below) |
| `--strategy` | Attack strategy ID(s): 0, 1 (see strategy table) |
| `--benchmark` | Benchmark: `rmc`, `mal`, `pku`, `total` (= rmc + mal), `trainset` |
| `--codelang` | Programming language: `py`, `cpp`, `java` (default: `py`) |
| `--run_rounds` | Number of experiment rounds (default: 1) |
| `--num_shards` / `--shard_id` | Data-parallel sharding for multi-GPU runs |
| `--quiet_generation` | Suppress per-token generation output |

**Model index mapping:**

| Index | Model |
|-------|-------|
| 0 | Meta-Llama-3-8B-Instruct |
| 1 | Qwen2.5-7B-Instruct |
| 2 | Qwen2.5-Coder-7B-Instruct |

> Add defended models (e.g., `Qwen2.5-Coder-7B-Instruct-defended`) to `model_info_dict` and `run_parallel.sh`'s `MODEL_DIR_MAP` after training.

### Attack: Multi-GPU Parallel Evaluation

`run_parallel.sh` automates 8-GPU sharded evaluation with automatic result merging. Run from the project root:

```bash
# Evaluate all models (0-2) with strategies 0,1 on rmc+mal benchmarks
RUN_MODEL_INDEX="0 1 2" STRATEGY="0 1" BENCHMARK=total bash Attack/LocalModels/script/run_parallel.sh
```

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `GPU_IDS` | `0,1,2,3,4,5,6,7` | Comma-separated GPU IDs to use |
| `RUN_MODEL_INDEX` | `0 1 2` | Model indices (space or comma separated) |
| `STRATEGY` | `0 1` | Strategy IDs |
| `BENCHMARK` | `total` | `rmc`, `mal`, `pku`, or `total` (= rmc + mal) |
| `CODELANG` | `py` | `py`, `cpp`, or `java` |
| `RUN_ROUNDS` | `1` | Number of experiment rounds |
| `QUIET_GENERATION` | `true` | Suppress per-token generation output |
| `MERGE_AFTER` | `true` | Auto-merge shards after completion |
| `MERGE_CLEANUP` | `true` | Remove shard temp files after merge |
| `LOG_DIR` | `logs_sharded` | Directory for per-shard log files |

The script handles:
1. Model-aware sharding (8/4/2/1 shards depending on model size)
2. Per-shard GPU assignment and parallel launch
3. Automatic result merging via `merge_shards.py` after all shards complete
4. Resume-friendly: re-run with same params skips already-completed shards

### Attack: Evaluate Results

```bash
cd Attack/LocalModels

# Evaluate Malicious Ratio (MR)
python script/evaluate_mr.py \
    --file_prefix 0 1 --run_model_indexs 0 1 2 --results_group total

# Evaluate Attack Success Rate (ASR)
python script/evaluate_asr.py \
    --file_prefix 0 1 --run_model_indexs 0 1 2 --results_group total
```

### Attack: Merge Sharded Results

```bash
cd Attack/LocalModels

# Merge shards for a specific strategy × benchmark × model
python script/merge_shards.py \
    --strategy 1 --benchmark rmc --model_dir Qwen2.5-Coder-7B-Instruct --cleanup

# Merge with explicit timestamp and prune duplicate run_info
python script/merge_shards.py \
    --strategy 1 --benchmark rmc \
    --model_dir Qwen2.5-Coder-7B-Instruct \
    --timestamp 20260509093627 --cleanup --prune-run-info
```

### Attack: API Model Evaluation

```bash
cd Attack/APIModels

# Run all models × benchmarks × methods (Vanilla + Ours)
python run.py

# Run a specific subset
python run.py --models gpt-5-mini gpt-5 --benchmarks mal --methods Ours

# Run multiple rounds for statistical reliability
python run.py --rounds 1 2 3 --workers 8
```

**Key arguments for `run.py`:**

| Argument | Description |
|----------|-------------|
| `--models` | Models to evaluate: `gpt-5-mini`, `gpt-5` (OpenAI); `minimax`, `gptoss`, `minimax25`, `gptossmini` (Fireworks). Default: all |
| `--benchmarks` | Benchmarks: `mal` (MalwareBench), `rmc` (RMCBench). Default: both |
| `--methods` | Attack methods: `Vanilla` (plain generation), `Ours` (grammar-constrained). Default: both |
| `--rounds` | Run round numbers (default: `1`). Multiple rounds improve statistical reliability |
| `--workers` | Number of parallel workers (default: `8`). Tune for API rate limits |

**Available API models:**

| Provider | Shorthand | Full Model |
|----------|-----------|------------|
| OpenAI | `gpt-5-mini` | GPT-5 Mini |
| OpenAI | `gpt-5` | GPT-5 |
| Fireworks | `minimax` | MiniMax-M2.7P |
| Fireworks | `gptoss` | GPT-OSS-120B |
| Fireworks | `minimax25` | MiniMax-M2.5 |
| Fireworks | `gptossmini` | GPT-OSS-20B |

After generation completes, evaluate the results with the same evaluation scripts used for LocalModels:

### Defense: CodeShield Data Construction

```bash
# Step 1, run `run_main.py` on trainset with target model under GCD

python Attack/LocalModels/script/run_main.py \
    --run_model_index 0 --strategy 1 --codelang py --benchmark trainset

# Step 2, evaluate the results

python Attack/LocalModels/script/evaluate_asr.py \
    --file_prefix 1 --run_model_indexs 0 --results_group trainset

# Step 3, constructing training data(remember download OCI dataset to `<path/to/sft_data.json>` and regularize it)

python Defense/build_training_data.py \
    --eval_result  <path/to/eval_result.xlsx> \
    --rejections   Attack/prompt/train/collected_prompts_2000.xlsx \
    --sft_source   OCI \
    --sft_data     <path/to/sft_data.json> \
    --output_dpo   <path/to/output_dpo.json> \
    --output_sft   <path/to/output_sft.json> \
    --k 1600 --j 5 --SFT 2.5 --seed 42
```

The script builds DPO preference pairs from evaluation results:
1. Identifies responses where the model failed to defend (verdict == BAD)
2. Constructs two-stage DPO pairs: (y_refuse > y_honeypot) and (y_honeypot > y_harmful)
3. Samples SFT data from OpenCodeInstruct for regularization
4. Outputs in LLaMA-Factory ShareGPT format

### Defense: CodeShield Training

All commands are run from the `Defense/` directory.

```bash
cd Defense

python train.py \
    --model_name_or_path ../Attack/LocalModels/models/Qwen2.5-Coder-7B-Instruct \
    --dpo_data          data/qwencoder7b/dpo.json \
    --sft_data          data/qwencoder7b/sft.json \
    --SFT               2.5 \
    --output_dir        saves/qwencoder7b \
    --lora_r            16 \
    --lora_alpha        32 \
    --learning_rate      1e-5 \
    --per_device_batch_size 4 \
    --gradient_accumulation_steps 4
```

This launches joint SFT+DPO interleaved training with LoRA. The trained adapter is saved to `saves/qwencoder7b/checkpoint-latest/`.

### Defense: Merge LoRA Adapter

After training, merge the LoRA adapter back into the base model using `llamafactory-cli export`:

```bash
cd Defense

# Merge Qwen2.5-Coder-7B adapter
llamafactory-cli export merge_config/merge_coder7b.yaml

# Merge Qwen2.5-7B adapter
llamafactory-cli export merge_config/merge_7b.yaml

# Merge Llama-3-8B adapter
llamafactory-cli export merge_config/merge_llama3.yaml
```

The merged (defended) model is exported to `../Attack/LocalModels/models/<model>-defended/` and can then be used in `run_main.py` for evaluation.

## 🎯 Attack Strategies

| ID | Strategy | Description |
|----|----------|-------------|
| 0 | Vanilla | Native generation |
| 1 | Ours | Grammar-constrained decoding, CodeSpear |

## 📊 Benchmarks

- **RMCBench**: Refuse Malicious Code Benchmark
- **MalwareBench**: Malicious code generation benchmark

## 📦 Dependencies

See `pyproject.toml` for full version constraints. Key packages:

| Package | Purpose |
|---------|---------|
| `torch>=2.10,<2.11` | Deep learning framework (CUDA 12.8) |
| `transformers` | HuggingFace model loading & inference |
| `llguidance` | Grammar-constrained decoding |
| `accelerate` | Multi-GPU device mapping |
| `llamafactory` | LoRA merge / model export |
| `pandas`, `openpyxl` | Data processing (xlsx prompts & results) |
| `openai`, `requests` | API model evaluation |

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.