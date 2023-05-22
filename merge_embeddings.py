import pandas as pd
import os
import argparse

def merge_emb_files(emb_directory, output_directory):


    file_list = sorted(os.listdir(emb_directory))
    emb_df = pd.read_pickle(emb_directory+ file_list[0], compression="gzip")
    output_file = output_directory + emb_directory.split("/")[-2] + ".gzip"

    print("merging embeddings-files...")
    for index, gzip_file in enumerate(file_list[1:]):
        current_file_path = emb_directory + gzip_file
        if index % 10 == 0:
            print(f"Merging file {gzip_file}.")
        current_df = pd.read_pickle(current_file_path, compression="gzip")
        new_df = pd.concat([emb_df,current_df], join="inner")
        emb_df = new_df.copy()

    print("Saving merged file...")
    emb_df.to_pickle(output_file, compression="gzip")
    print(f"File saved to: {output_file}.")

    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_directory', help='directory containing small files for merging')
    parser.add_argument('--output_directory', help='directory to save the obtained merged file')
    args = parser.parse_args()
    emb_dir = args.emb_directory
    out_dir = args.output_directory
    merge_emb_files(emb_dir, out_dir)

    # emb_dir = "./data/embeddings/english/" 
    # out_dir = "./data/embeddings/merged_embs/"
    # merge_emb_files(emb_dir, out_dir)
