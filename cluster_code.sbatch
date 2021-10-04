#!/usr/bin/env bash
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --constrain='balrog|belegost|gondor|khazadum|rivendell|shelob|shire|smaug'

source "/Midgard/home/areichlin/miniconda3/etc/profile.d/conda.sh"
conda activate env_torch

python3 /Midgard/home/areichlin/GBML_RL/main.py --exploiter_iteration ${EXPLO_IT} --explorer_loss ${EXPLO_LOSS} --sparse_reward ${SPARSE} --decoupled_optimization ${DEC_OPT}