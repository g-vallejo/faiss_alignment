
import pandas as pd
import numpy as np
from datasets import load_dataset 
from datetime import datetime

date = datetime.now().strftime("%Y_%m_%d_%H%M")

new_df_train = pd.DataFrame(columns=['context', 'answer', 'question', 'id'])
new_df_dev = pd.DataFrame(columns=['context', 'answer', 'question', 'id'])

lang_data = {
    "es": {
        "dataset": "mlqa",
        "filename": "mlqa-translate-train.es",
        "split": "train",
        "samples": 35000
    },

    "de": {
        "dataset": "mlqa",
        "filename": "mlqa-translate-train.de",
        "split": "train",
        "samples": 35000
    },



    "en": {
        "dataset": "squad",
        "filename": "mlqa.en.en",
        "split": "train",
        "samples": 35000
    }
}

for lang, features in lang_data.items():
    if lang == "en":
        loaded_data = load_dataset(features["dataset"], #features["filename"],
                               cache_dir="/scratch/punim0478/gvallejo/MT5/.cache/")
    else:
        loaded_data = load_dataset(features["dataset"], features["filename"],
                               cache_dir="/scratch/punim0478/gvallejo/MT5/.cache/")

    df = pd.DataFrame(data=loaded_data[features["split"]], columns=loaded_data[features["split"]].column_names)
    df['answer'] = df['answers'].apply(lambda x: x['text'][0])
    df_sample = df[['context', 'answer', 'question', 'id']].sample(n=features["samples"], random_state=257)
    train_len = features["samples"] * 0.02
    new_df_train = pd.concat([new_df_train, df_sample[:-int(train_len)]], ignore_index=True) # We take 10% for dev
    print(f"Train for {lang} comprises {df_sample[:-int(train_len)].count()}, current new dataset lenght is {new_df_train.count()}")
    new_df_dev = pd.concat([new_df_dev, df_sample[-int(train_len):]], ignore_index=True) # We take 10% for dev
    print(f"Dev for {lang} comprises {df_sample[-int(train_len):].count()}, current new dataset lenght is {new_df_dev.count()}")

print("Data generated, saving...")

new_df_train.to_csv(f"./data/qa_train_data_de_en_es_{date}.csv", sep="\t", index=None) 
new_df_dev.to_csv(f"./data/qa_dev_data_de_en_es_{date}.csv", sep="\t", index=None)

print("Saved data!")
 
