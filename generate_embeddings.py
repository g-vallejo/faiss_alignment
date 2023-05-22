
from sentence_transformers import SentenceTransformer, models
import argparse
import pandas as pd


def generate_embeddings(model, df, data, column, output_dir):

    step = 10000

    for i in range(0, len(df), step):
        columns = df.columns
        sub_df = df[i:i+step][columns].copy()
        
        # Generate embeddings to store in the sub-df
        embeddings = model.encode(sub_df[column].to_list(), show_progress_bar=True, convert_to_numpy=True)
        sub_df["embeddings"] = embeddings.tolist()
        sub_df.to_pickle(f"{output_dir}{data}_{i}-{i+len(sub_df)-1}.gzip", compression="gzip", protocol=4)


def main(csv_file, data, column="context", output_dir="./"):

    df = pd.read_csv(csv_file, sep="\t") 
    model_name = 'LaBSE'
    model_labse = SentenceTransformer(model_name)
    generate_embeddings(model_labse, df, data, column, output_dir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', help='csv file containing data')
    parser.add_argument('--output_dir', help='directory to save data')
    parser.add_argument('--column_name', help='colunm in the csv to use')
    parser.add_argument('--data_name', help='data name to put in output file name')
    args = parser.parse_args()
    print(args)
    csv = args.csv
    column = args.column_name
    data_name = args.data_name
    outdir = args.output_dir
    main(csv, data_name, column, outdir)
