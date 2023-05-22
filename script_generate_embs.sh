#!/bin/bash

#SBATCH -q gpgpudeeplearn
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=10G
#SBATCH --partition=deeplearn
#SBATCH -A punim0478
#SBATCH --gres=gpu:A100
#SBATCH --qos=gpgpudeeplearn
#SBATCH --job-name="faiss_en"

#SBATCH --mail-user=gvallejo@student.unimelb.edu.au
#SBATCH --mail-type=ALL
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

source /scratch/punim0478/gvallejo/miniconda3/bin/activate sentbert
export TRANSFORMERS_CACHE=/scratch/punim0478/gvallejo/qa/data/.cache/

which python


srun python generate_embeddings.py --csv ../QuestEval/examples/data/Eng_Spa_20220328_final_without2daydelta.tsv \
    --output_dir ../QuestEval/examples/data/ \
    --data_name reuters_es_full \
    --column news_text_EN 
