import faiss                   # make faiss available
import numpy as np
import pandas as pd
import argparse

# df_1 = pd.read_pickle("./data/embeddings/merged_embs/english.gzip", compression="gzip")
# df_2= pd.read_pickle("./data/embeddings/merged_embs/german.gzip", compression="gzip")

def search_candidates(database_file, query_file, output_file):

    db_df = pd.read_pickle(database_file, compression="gzip")
    query_df = pd.read_pickle(query_file, compression="gzip")
    xb = np.stack(db_df["embeddings"].to_numpy()).astype('float32')
    xq = np.stack(query_df["embeddings"].to_numpy()).astype('float32')



    index = faiss.IndexFlatL2(len(xb[0]))   # build the index
    print(index.is_trained)
    index.add(xb)                  # add vectors to the index
    print(index.ntotal)

    k = 4                          # we want to see 4 nearest neighbors
    D, I = index.search(xq, k)
    
    print("Generating the output file.")
#     faiss_df = pd.DataFrame(data=I, columns=["Doc1", "Doc2", "Doc3", "Doc4", "Doc5", "Doc6", "Doc7", "Doc8"]) 
#     faiss_df[["Score1", "Score2", "Score3", "Score4", "Score5", "Score6", "Score7", "Score8"]] = D

    faiss_df = pd.DataFrame(data=I, columns=["Doc1", "Doc2", "Doc3", "Doc4"]) 
    faiss_df[["Score1", "Score2", "Score3", "Score4"]] = D  
  
    print("Output is ready, saving...") 
    faiss_df.to_pickle(output_file, compression="gzip", protocol=4)
    print(f"Output saved to {output_file}") 

    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_file', help='pickle file contaning the embeddings of docs to search in.')
    parser.add_argument('--query_file', help='pickle file contaning the embeddings of docs to search for.')
    parser.add_argument('--output_file', help='pickle file to save results of the search.') 
    args = parser.parse_args()
    db_file = args.database_file
    q_file = args.query_file
    out_file = args.output_file
    search_candidates(db_file, q_file, out_file)
