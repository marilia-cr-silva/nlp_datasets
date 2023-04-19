"""
@inproceedings{wang-2017-liar,
    title = "{``}Liar, Liar Pants on Fire{''}: A New Benchmark Dataset for Fake News Detection",
    author = "Wang, William Yang",
    booktitle = "Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2017",
    address = "Vancouver, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P17-2067",
    doi = "10.18653/v1/P17-2067",
    pages = "422--426",
}
"""

# %% loading libraries

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import os
import re
import html
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore")

# %% function to reduce the noise

def noise_mitigation(string: str) -> str:
    """
    this function returns a pre-processed string.
    input - param:
    string:
    the text that is going to be processed
    output:
    string (str):
    pre-processed string
    """
    new_string = string.split("\n")
    string = " ".join(new_string)
    string = re.sub("\s\#\s|\@user\s?|Says\s","",string)
    string = re.sub("\-\-+|\s\-\s"," ",string)
    string = re.sub("\s?\@\s"," at ",string)
    string = re.sub(r"(https?:(\/\/))?(www\.)?(\w+)\.(\w+)(\.?\/?)+([a-zA-Z0–9@:%&._\+-?!~#=]*)?(\.?\/)?([a-zA-Z0–9@:%&._\/+-~?!#=]*)?(\.?\/)?"," ",string) # websites
    string = re.sub('\s?http(s)?\:\/+.*','',string) # shortened url    
    string = re.sub("w\/|W\/","with",string)
    string = re.sub("\\&amp;"," and ",string)
    
    string = html.escape(string)
    string = html.unescape(string)
    string = BeautifulSoup(string, "lxml")
    string = string.get_text()

    string = re.sub("aa+","aa",string)
    string = re.sub("bb+","bb",string)
    string = re.sub("cc+","cc",string)
    string = re.sub("dd+","dd",string)
    string = re.sub("ee+","ee",string)
    string = re.sub("ff+","ff",string)
    string = re.sub("gg+","gg",string)
    string = re.sub("hh+","hh",string)
    string = re.sub("ii+","ii",string)
    string = re.sub("jj+","j",string)
    string = re.sub("ll+","ll",string)
    string = re.sub("mm+","mm",string)
    string = re.sub("nn+","nn",string)
    string = re.sub("oo+","oo",string)
    string = re.sub("pp+","pp",string)
    string = re.sub("qq+","q",string)
    string = re.sub("rr+","rr",string)
    string = re.sub("ss+","ss",string)
    string = re.sub("tt+","tt",string)
    string = re.sub("uu+","uu",string)
    string = re.sub("vv+","vv",string)
    string = re.sub("ww+","ww",string)
    string = re.sub("xx+","xx",string)
    string = re.sub("yy+","yy",string)
    string = re.sub("zz+","zz",string)

    string = re.sub("AA+","AA",string)
    string = re.sub("BB+","BB",string)
    string = re.sub("CC+","CC",string)
    string = re.sub("DD+","DD",string)
    string = re.sub("EE+","EE",string)
    string = re.sub("FF+","FF",string)
    string = re.sub("GG+","GG",string)
    string = re.sub("HH+","HH",string)
    string = re.sub("II+","II",string)
    string = re.sub("JJ+","JJ",string)
    string = re.sub("KK+","KK",string)
    string = re.sub("LL+","LL",string)
    string = re.sub("MM+","MM",string)
    string = re.sub("NN+","NN",string)
    string = re.sub("OO+","OO",string)
    string = re.sub("PP+","PP",string)
    string = re.sub("QQ+","QQ",string)
    string = re.sub("RR+","RR",string)
    string = re.sub("SS+","SS",string)
    string = re.sub("TT+","TT",string)
    string = re.sub("UU+","UU",string)
    string = re.sub("VV+","VV",string)
    string = re.sub("WW+","WW",string)
    string = re.sub("XX+","XX",string)
    string = re.sub("YY+","YY",string)
    string = re.sub("ZZ+","ZZ",string)

    string = re.sub("ha(ha)+","haha",string)
    string = re.sub("he(he)+","hehe",string)
    string = re.sub("hi(hi)+","hihi",string)
    string = re.sub("HA(HA)+","HAHA",string)
    string = re.sub("HE(HE)+","HEHE",string)
    string = re.sub("HI(HI)+","HIHI",string)

    string = re.sub("\s\.",".",string)
    string = re.sub('\"{2,}','"',string)
    string = re.sub("\s\!","!",string)
    string = re.sub("\s\?","?",string)
    string = re.sub("\s\,",",",string)
    string = re.sub('^"+|^\'+|"+$|\'+$','',string)
    string = re.sub('^"+|^\'+|"+$|\'+$','',string)
    string = re.sub('"+','"',string)

    string = re.sub(r"[а-яА-Я]","",string) # Cyrillic characters [\u4000-\u04ff]
    string = re.sub(r"[\u4e00-\u9fff]+","",string) # Chinese characters
    string = re.sub(r"[\u0621-\u064a\ufb50-\ufdff\ufe70-\ufefc]","",string) # Arabic Characters
    string = re.sub("\(\/\S+\)","",string)
    string = re.sub("\[|\]|\{|\}|\(|\)|\>|\<|\=|\_","",string) # e.g., [](){}

    new_string = string.split()
    string = " ".join(new_string)
    
    return string


# %% loading dataset and creating dataframes
dataset = load_dataset("liar")
df_test_aux = pd.DataFrame([dataset["test"]["statement"],dataset["test"]["label"]]).T
df_validation_aux = pd.DataFrame([dataset["validation"]["statement"],dataset["validation"]["label"]]).T
df_test = pd.concat([df_validation_aux,df_test_aux])
df_test.columns = ["text","label"]
df_train = pd.DataFrame([dataset["train"]["statement"],dataset["train"]["label"]]).T
df_train.columns = ["text","label"]

df_train["new_text"] = df_train["text"].apply(lambda x: noise_mitigation(x))
df_train = df_train.drop_duplicates(subset=["new_text"],keep="first")
df_train = df_train[["new_text","label"]]
df_train.rename(columns={"new_text": "text"}, inplace=True)
df_train = df_train.sample(frac=1,random_state=42).reset_index(drop=True)

df_test["new_text"] = df_test["text"].apply(lambda x: noise_mitigation(x))
df_test = df_test.drop_duplicates(subset=["new_text"],keep="first")
df_test = df_test[["new_text","label"]]
df_test.rename(columns={"new_text": "text"}, inplace=True)
df_test = df_test.sample(frac=1,random_state=42).reset_index(drop=True)


# %%

df_train.to_csv(f"fn_01_multi_train.csv",sep=";",index=False)
df_test.to_csv(f"fn_01_multi_test.csv",sep=";",index=False)

# %%
unique_classes = sorted(df_train["label"].unique())

list_csv = []
for i in tqdm(range(len(unique_classes))):
    for j in range(i+1,len(unique_classes)):
        # train
        df_aux = df_train.loc[(df_train["label"] == unique_classes[i]) | (df_train["label"] == unique_classes[j])]
        df_aux.to_csv(f"fn_01_bin_train_{i}_{j}.csv",sep=";",index=False)
        list_csv.append([f"fn_01_bin_train_{i}_{j}.csv",f"{unique_classes[i],unique_classes[j]}"])
        # test
        df_aux = df_test.loc[(df_test["label"] == unique_classes[i]) | (df_test["label"] == unique_classes[j])]
        df_aux.to_csv(f"fn_01_bin_test_{i}_{j}.csv",sep=";",index=False)
        list_csv.append([f"fn_01_bin_test_{i}_{j}.csv",f"{unique_classes[i],unique_classes[j]}"])

df_list_csv = pd.DataFrame(list_csv,columns=["file_name","classes"])
df_list_csv.to_csv("fn_01_bin_explained.csv",sep=";",index=False)

number_classes = unique_classes
explained = {i: unique_classes[i] for i in range(number_classes)}