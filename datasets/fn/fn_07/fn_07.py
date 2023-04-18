"""
@inproceedings{Thorne19FEVER2,
    title = {\href{http://dx.doi.org/10.18653/v1/D19-6601}{The {FEVER2.0} Shared Task}},
    author = "Thorne, James  and
      Vlachos, Andreas  and
      Cocarascu, Oana  and
      Christodoulopoulos, Christos  and
      Mittal, Arpit",
    booktitle = "Proceedings of the Second Workshop on Fact Extraction and VERification (FEVER)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-6601",
    doi = "10.18653/v1/D19-6601",
    pages = "1--6",
    abstract = "We present the results of the second Fact Extraction and VERification (FEVER2.0) Shared Task. The task challenged participants to both build systems to verify factoid claims using evidence retrieved from Wikipedia and to generate adversarial attacks against other participant{'}s systems. The shared task had three phases: \textit{building, breaking and fixing}. There were 8 systems in the builder{'}s round, three of which were new qualifying submissions for this shared task, and 5 adversaries generated instances designed to induce classification errors and one builder submitted a fixed system which had higher FEVER score and resilience than their first submission. All but one newly submitted systems attained FEVER scores higher than the best performing system from the first shared task and under adversarial evaluation, all systems exhibited losses in FEVER score. There was a great variety in adversarial attack types as well as the techniques used to generate the attacks, In this paper, we present the results of the shared task and a summary of the systems, highlighting commonalities and innovations among participating systems.",
}
"""

import pandas as pd
import json
import os
from tqdm import tqdm

dataset_name = 'fn_07'
train_filename = 'fever2-fixers-dev.jsonl'

# Garantindo que os diretorios de split para teste e treino existem
folders = ['train_split']
files_in_curr_dir = os.listdir()
for folder in folders:
    if folder not in files_in_curr_dir:
        os.mkdir(folder)

content = []
with open(train_filename, 'r') as f:
    content = f.readlines()

lines = [json.loads(c) for c in content]

df_content = pd.DataFrame(lines)
df_content = df_content[["claim", "label"]]
df_content.rename({'claim': 'text'}, axis=1, inplace=True)
df_content['label'] = df_content['label'].apply(lambda x: x.upper())

# %%
df_train, df_test = train_test_split(
    df_content, test_size=0.3, random_state=42,shuffle = True)

df_train.to_csv(f"fn_07_multi_train.csv",sep=";",index=False)
df_test.to_csv(f"fn_07_multi_test.csv",sep=";",index=False)

# %%
unique_classes = sorted(df_train["label"].unique())

list_csv = []
for i in tqdm(range(len(unique_classes))):
    for j in range(i+1,len(unique_classes)):
        # train
        df_aux = df_train.loc[(df_train["label"] == unique_classes[i]) | (df_train["label"] == unique_classes[j])]
        df_aux.to_csv(f"fn_07_bin_train_{i}_{j}.csv",sep=";",index=False)
        list_csv.append([f"fn_07_bin_train_{i}_{j}.csv",f"{unique_classes[i],unique_classes[j]}"])
        # test
        df_aux = df_test.loc[(df_test["label"] == unique_classes[i]) | (df_test["label"] == unique_classes[j])]
        df_aux.to_csv(f"fn_07_bin_test__{i}_{j}.csv",sep=";",index=False)
        list_csv.append([f"fn_07_bin_test__{i}_{j}.csv",f"{unique_classes[i],unique_classes[j]}"])

df_list_csv = pd.DataFrame(list_csv,columns=["file_name","classes"])
df_list_csv.to_csv("fn_07_bin_explained.csv",sep=";",index=False)