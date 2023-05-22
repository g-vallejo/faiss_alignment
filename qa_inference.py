from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset, load_metric
import pandas as pd
import numpy as np
import torch
from bert_score import score
import json
import argparse

from datetime import datetime
date = datetime.now().strftime("%Y_%m_%d_%H%M")

def evaluate_mt5_qg(tokenizer_cp, model_cp, eval_data_tsv, rouge=True, em=True, bs=False, save_gens=False):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_cp)
    print("loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_cp).to("cuda")
    print("model loaded!")
    rouge = load_metric("rouge")
    exact_match = load_metric("exact_match")

    #read data
    data = pd.read_csv(eval_data_tsv, sep="\t")
    prefix = "answer: "
    raw_input = [prefix + doc for doc in data["q_e_context"]]
    labels = data["answer"]
    print("data ready")
    data_name = eval_data_tsv.split("/")[-1].replace(".csv", "")
#    # inference
    pred_answers = []
    for sample in raw_input:
        input_ids = tokenizer(sample, return_tensors="pt").input_ids.to("cuda")  # Batch size 1
        outputs = model.generate(input_ids)
        decoded_pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred_answers.append(decoded_pred) 

    print("all qs answered.")
    generated_data = {"pred_answers": pred_answers, "gold_ans": labels}
    
    # save data 
    if save_gens:
        gen_df = pd.DataFrame(data=generated_data)
        output_pred = f"./data/predictions_{data_name}_{date}.csv" 
        gen_df.to_csv(output_pred, sep="\t", index=None)
        print("predictions saved to {output_pred}")
    decoded_preds = pred_answers 
    # decoded_labels = ["Wie viele Punkte gab die Verteidigung der Panthers ab?"]
    decoded_labels = labels #["Wie viele Punkte hat die Verteidigung der Panthers abgegeben?"]

    # metrics
    result = {}
    if rouge:
        resulted_rouge = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result["rouge"] = resulted_rouge
        #result = {key: value.mid.fmeasure * 100 for key, value in resulted_rouge.items()}
        print("done with rouge!")
    if em:
        resulted_em = exact_match.compute(predictions=decoded_preds, references=decoded_labels, ignore_case=True, ignore_punctuation=True)
        result["exact_match"] = resulted_em["exact_match"]
        print("EM done!")
    if bs:
        print("Calculating BS")
        bs_precision, bs_recall, bs_f1 = score(gen_df["pred_answers"].to_list(), gen_df["gold_ans"].to_list(), model_type="distilbert-base-multilingual-cased", num_layers=5, device="cuda") 
        result["bs_p"] = bs_precision.detach().numpy().tolist()
        result["bs_r"] = bs_recall.detach().numpy().tolist()
        result["bs_f1"] = bs_f1.detach().numpy().tolist()
        print("Done with BS!")
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in decoded_preds]
    result["gen_len"] = np.mean(prediction_lens)
    
#   # result["bertscore"] = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="es")
    # pd.DataFrame(data=list({k: round(v, 4) for k, v in result.items()})).to_csv(f"./data/evaluation_{date}.csv", sep="\t", index=None) 
    with open(f"./data/evaluation_{date}.txt", 'w') as jfile:
        jfile.write(json.dumps(result))
    print(result)
    return result.items() #{k: round(v, 4) for k, v in result.items()}
#    return "test"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='pickle file contaning the embeddings of docs to search in.')
    parser.add_argument('--data_tsv_file', help='pickle file contaning the embeddings of docs to search for.')
    args = parser.parse_args()
    cp_path = args.model_path
    tsv_file = args.data_tsv_file
    # cp_path = "google/mt5-small"
    # cp_path = "../MT5/experiments/google/mt5-base_qa-ml/2023_04_11_2155/checkpoint-255000"
    # cp_path = "experiments/run4/checkpoint-50000"
    # tsv_file = "./data/xquad.es.tsv"
    # tsv_file = "data/finetune_dev_data_noneg.csv"
    print(evaluate_mt5_qg(cp_path, cp_path, tsv_file, rouge=True, em=True, bs=True, save_gens=True))


