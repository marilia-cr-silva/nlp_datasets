"""
@misc{trip_advisor_2018,
  doi = {10.5281/ZENODO.1219899},
  url = {https://zenodo.org/record/1219899},
  author = {{Barkha Bansal}},
  title = {Tripadvisor Hotel Review Dataset},
  publisher = {Zenodo},
  year = {2018},
  copyright = {Creative Commons Attribution-NonCommercial 4.0}
}
"""

# %% loading libraries
import csv
import html
import io
import re
import warnings

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings("ignore")


# %% function to reduce the noise


def noise_mitigation(aux):
    string = str(aux)
    new_string = string.split("\n")
    string = " ".join(new_string)
    string = re.sub(r"\n|\t|\\n|\\t", "", string)
    string = re.sub(
        "\s\#\s|\@user\s?|\s\#\s|\@USER\s?|Says\s|\!+\sRT\s|\s?RT\s", "", string
    )
    string = re.sub("\-\-+|\s\-\s", " ", string)
    string = re.sub("\s?\@\s", " at ", string)
    string = re.sub(
        r"(https?:(\/\/))?(www\.)?(\w+)\.(\w+)(\.?\/?)+([a-zA-Z0–9@:%&._\+-?!~#=]*)?(\.?\/)?([a-zA-Z0–9@:%&._\/+-~?!#=]*)?(\.?\/)?",
        " ",
        string,
    )  # websites
    string = re.sub("\s?http(s)?\:\/+.*|HTTPS?", "", string)  # shortened url
    string = re.sub("w\/|W\/", "with", string)
    string = re.sub("\\&amp;", " and ", string)
    string = re.sub("&", " and ", string)
    string = re.sub("\sPage\s\d{1,2}\:|©(\s*\S*)*", "", string)
    string = re.sub("\@\w+", "@user", string)
    string = re.sub("^,\s?|^\.\s?|^:|\||《|》|•|↝||¤|ï|ð|«|☀|“", "", string)
    string = re.sub(
        "\d{2}\:\d{2}"
        "|\,?\s?(\d{1,2})?\s?\S{3,20}\s\d{4}\s?(UTC)?"
        "|\S{3,20}\s\d{1,2}\,\s\d{4}\s?(UTC)?"
        "|\d{2}\/\d{2}\/\d{2,4}",
        "",
        string,
    )
    string = re.sub("\W+'|'\W+", "'", string)

    string = html.escape(string)
    string = html.unescape(string)
    string = BeautifulSoup(string, "lxml")
    string = string.get_text()

    new_string = string.split('"')
    string = "".join(new_string)

    string = re.sub("bb+", "bb", string)
    string = re.sub("cc+", "cc", string)
    string = re.sub("dd+", "dd", string)
    string = re.sub("ff+", "ff", string)
    string = re.sub("gg+", "gg", string)
    string = re.sub("hh+", "hh", string)
    string = re.sub("jj+", "j", string)
    string = re.sub("kk+", "kk", string)
    string = re.sub("ll+", "ll", string)
    string = re.sub("mm+", "mm", string)
    string = re.sub("nn+", "nn", string)
    string = re.sub("pp+", "pp", string)
    string = re.sub("qq+", "q", string)
    string = re.sub("rr+", "rr", string)
    string = re.sub("ss+", "ss", string)
    string = re.sub("tt+", "tt", string)
    string = re.sub("vv+", "vv", string)
    string = re.sub("ww+", "ww", string)
    string = re.sub("xx+", "xx", string)
    string = re.sub("yy+", "yy", string)
    string = re.sub("zz+", "zz", string)

    string = re.sub("BB+", "BB", string)
    string = re.sub("CC+", "CC", string)
    string = re.sub("DD+", "DD", string)
    string = re.sub("FF+", "FF", string)
    string = re.sub("GG+", "GG", string)
    string = re.sub("HH+", "HH", string)
    string = re.sub("JJ+", "JJ", string)
    string = re.sub("KK+", "KK", string)
    string = re.sub("LL+", "LL", string)
    string = re.sub("MM+", "MM", string)
    string = re.sub("NN+", "NN", string)
    string = re.sub("PP+", "PP", string)
    string = re.sub("QQ+", "QQ", string)
    string = re.sub("RR+", "RR", string)
    string = re.sub("SS+", "SS", string)
    string = re.sub("TT+", "TT", string)
    string = re.sub("VV+", "VV", string)
    string = re.sub("WW+", "WW", string)
    string = re.sub("XX+", "XX", string)
    string = re.sub("YY+", "YY", string)
    string = re.sub("ZZ+", "ZZ", string)

    string = re.sub("ha(ha)+", "haha", string)
    string = re.sub("he(he)+", "hehe", string)
    string = re.sub("hi(hi)+", "hihi", string)
    string = re.sub("HA(HA)+", "HAHA", string)
    string = re.sub("HE(HE)+", "HEHE", string)
    string = re.sub("HI(HI)+", "HIHI", string)

    string = re.sub("\.\.+", "...", string)
    string = re.sub("\s\.", ".", string)
    string = re.sub('"{2,}', '"', string)
    string = re.sub("\s\!", "!", string)
    string = re.sub("\s\?", "?", string)
    string = re.sub("\s\,", ",", string)
    string = re.sub("^\"+|^'+|\"+$|'+$", "", string)
    string = re.sub(
        "^\"+|^'+|\"+$|'+$", "", string
    )  # if it has several types of quotations in the beginning
    string = re.sub('"+', '"', string)

    # Cyrillic characters [\u4000-\u04ff]
    string = re.sub(r"[а-яА-Я]", "", string)
    string = re.sub(r"[\u4e00-\u9fff]+", "", string)  # Chinese characters
    string = re.sub(
        r"[\u0621-\u064a\ufb50-\ufdff\ufe70-\ufefc]", "", string
    )  # Arabic Characters
    string = re.sub("\(\/\S+\)", "", string)
    string = re.sub("\[|\]|\{|\}|\(|\)|\>|\<|\*|\=|\_",
                    "", string)  # e.g., [](){}
    string = re.sub(
        "^\"+|^'+|\"+$|'+$", "", string
    )  # if it has several types of quotations in the beginning
    string = re.sub("â€™", "'", string)

    string = re.sub('^:|^!|^\?|^\-|^\.|^"|^\/|^\\|$"', "", string)
    new_string = string.split()
    string = " ".join(new_string)
    string = re.sub('^:|^!|^\?|^\-|^\.|^"|^\/|^\\|$"', "", string)
    new_string = string.split()
    string = " ".join(new_string)
    new_string = string.split()
    string = " ".join(new_string)

    if len(string) < 3:
        string = ""

    return string


# %%
# %%

df_content = pd.read_csv("ua_06.csv")
df_content.columns = ["index", "text", "label"]
df_content = df_content[["text", "label"]]
df_content["text"] = df_content["text"].apply(lambda x: noise_mitigation(x))
df_content.assign(text=lambda df: df["text"].replace("", np.nan)).dropna().reset_index(
    drop=True
)
df_content = df_content.drop_duplicates(subset=["text"], keep="first")
df_content = df_content[["text", "label"]]
df_content = df_content.sample(frac=1, random_state=42).reset_index(drop=True)

df_train, df_test = train_test_split(
    df_content, test_size=0.3, random_state=42, shuffle=True
)
# %%

df_train.to_csv("ua_06_multi_train.csv", sep=";", index=False)
df_test.to_csv("ua_06_multi_test.csv", sep=";", index=False)

# %%
unique_classes = sorted(df_train["label"].unique())

for i in tqdm(range(len(unique_classes))):
    for j in range(i + 1, len(unique_classes)):
        # train
        df_aux = df_train.loc[
            (df_train["label"] == unique_classes[i])
            | (df_train["label"] == unique_classes[j])
        ]
        df_aux.to_csv(f"ua_06_bin_train_{i}_{j}.csv", sep=";", index=False)

        # test
        df_aux = df_test.loc[
            (df_test["label"] == unique_classes[i])
            | (df_test["label"] == unique_classes[j])
        ]
        df_aux.to_csv(f"ua_06_bin_test_{i}_{j}.csv", sep=";", index=False)

# %% saving explained.csv
number_classes = len(unique_classes)
explained_df = pd.DataFrame(
    {"index": range(number_classes), "label": list(unique_classes)}
)
explained_df.to_csv("ua_06_explained.csv", sep=";", index=False)
