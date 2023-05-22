#!/bin/bash

#SBATCH -q gpgpudeeplearn
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --partition=feit-gpu-a100
#SBATCH -A punim0478
#SBATCH --gres=gpu:A100
#SBATCH --qos=feit
#SBATCH --job-name="faiss_en"

#SBATCH --mail-user=gvallejo@student.unimelb.edu.au
#SBATCH --mail-type=ALL
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

source /scratch/punim0478/gvallejo/miniconda3/bin/activate sentbert
export TRANSFORMERS_CACHE=/scratch/punim0478/gvallejo/qa/data/.cache/

lang=en
which python


srun python faiss_search.py --database_file ./data/lang_embs/qa_dev_${lang}_embs.gzip \
    --query_file ./data/lang_embs/qa_dev_${lang}_embs.gzip \
    --output_file ./data/dev_${lang}_alignment_scores_4nn_20230425.gzip

