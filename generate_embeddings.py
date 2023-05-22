
from sentence_transformers import SentenceTransformer, models
import argparse
import pandas as pd


def generate_embeddings(model, df, output_dir):

    step = 10000

    for i in range(0, len(df), step):
        columns = df.columns
        sub_df = df[i:i+step][columns].copy()
        
        # Generate embeddings to store in the sub-df
        embeddings = model.encode(sub_df["context"].to_list(), show_progress_bar=True, convert_to_numpy=True)
        sub_df["embeddings"] = embeddings.tolist()
        sub_df.to_pickle(f"{output_dir}qa_training_{i}-{i+len(sub_df)-1}.gzip", compression="gzip", protocol=4)


def main(csv_file, output_dir="./"):

    df = pd.read_csv(csv_file, sep="\t") 
    model_name = 'LaBSE'
    model_labse = SentenceTransformer(model_name)
    generate_embeddings(model_labse, df, output_dir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', help='csv file containing data')
    parser.add_argument('--output_dir', help='directory to save data')
    args = parser.parse_args()
    csv = args.csv
    outdir = args.output_dir
    main(csv, outdir)
