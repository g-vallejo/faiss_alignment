#!/bin/bash

source /scratch/punim0478/gvallejo/miniconda3/bin/activate sentbert

python faiss_search.py --database_file ./data/merged_embs/small_embs.gzip \
  --query_file ./data/merged_embs/small_embs.gzip \
  --output_file ./data/qa_alignment_scores.gzip

