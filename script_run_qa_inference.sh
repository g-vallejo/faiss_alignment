#!/bin/bash

#SBATCH -q gpgpudeeplearn
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --partition=feit-gpu-a100
#SBATCH -A punim0478
#SBATCH --gres=gpu:A100
#SBATCH --qos=feit
#SBATCH --job-name="ft-qa_ml"

#SBATCH --mail-user=gvallejo@student.unimelb.edu.au
#SBATCH --mail-type=ALL
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

source /scratch/punim0478/gvallejo/miniconda3/bin/activate mt5_env 
export TRANSFORMERS_CACHE=/scratch/punim0478/gvallejo/MT5/.cache/
model_cp=/scratch/punim0478/gvallejo/MT5/experiments/qa_checkpoint/mt5-base_qa-ml/2023_04_11_2155/checkpoint-255000
ll $model_cp
which python

srun python qa_inference.py --model_path $model_cp --data_tsv_file ./data/xquad.en.qec.csv 

