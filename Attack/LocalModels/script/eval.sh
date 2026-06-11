#!/bin/bash
# Example evaluation commands

cd Attack/LocalModels/script

python evaluate_mr.py --file_prefix 0 1 --run_model_indexs 0 1 2 --results_group total

python evaluate_asr.py --file_prefix 0 1 --run_model_indexs 0 1 2 --results_group total