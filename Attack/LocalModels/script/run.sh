#!/bin/bash
# Example run commands for local model evaluation

cd Attack/LocalModels/script

CUDA_VISIBLE_DEVICES=0 python -u run_main.py --run_model_index 0 --strategy 0 1 --codelang py --benchmark total