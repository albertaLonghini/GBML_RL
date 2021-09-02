#!/usr/bin/env bash
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --constrain='balrog|belegost|gondor|khazadum|rivendell|shelob|shire'

source "/Midgard/home/longhini/miniconda3/etc/profile.d/conda.sh"
conda activate python3_env

python3 /Midgard/home/longhini/GBML_RL/RL_Project/rl_project.py --agent ${A} --mbs ${MBS}