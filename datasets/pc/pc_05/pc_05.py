"""
@inproceedings{rosenthal-etal-2017-semeval,
    title = "{S}em{E}val-2017 Task 4: Sentiment Analysis in {T}witter",
    author = "Rosenthal, Sara  and
      Farra, Noura  and
      Nakov, Preslav",
    booktitle = "Proceedings of the 11th International Workshop on Semantic Evaluation ({S}em{E}val-2017)",
    month = aug,
    year = "2017",
    address = "Vancouver, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/S17-2088",
    doi = "10.18653/v1/S17-2088",
    pages = "502--518",
    abstract = "This paper describes the fifth year of the Sentiment Analysis in Twitter task. SemEval-2017 Task 4 continues with a rerun of the subtasks of SemEval-2016 Task 4, which include identifying the overall sentiment of the tweet, sentiment towards a topic with classification on a two-point and on a five-point ordinal scale, and quantification of the distribution of sentiment towards a topic across a number of tweets: again on a two-point and on a five-point ordinal scale. Compared to 2016, we made two changes: (i) we introduced a new language, Arabic, for all subtasks, and (ii) we made available information from the profiles of the Twitter users who posted the target tweets. The task continues to be very popular, with a total of 48 teams participating this year.",
}

This file uses only the subtask A. In addition,
the tweets were retrieved before being preprocessed.
"""
# %% loading libraries
import gc
import html
import os
import re
import warnings
from io import StringIO
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

warnings.filterwarnings("ignore")

# %% function to reduce the noise


def noise_mitigation(aux):

    string = str(aux)
    new_string = string.split('\n')
    string = ' '.join(new_string)
    string = re.sub(r'\n|\t|\\n|\\t', '', string)
    string = re.sub(
        '\s\#\s|\@user\s?|\s\#\s|\@USER\s?|Says\s|\!+\sRT\s|\s?RT\s', '', string)
    string = re.sub('\-\-+|\s\-\s', ' ', string)
    string = re.sub('\s?\@\s', ' at ', string)
    string = re.sub(
        r'(https?:(\/\/))?(www\.)?(\w+)\.(\w+)(\.?\/?)+([a-zA-Z0–9@:%&._\+-?!~#=]*)?(\.?\/)?([a-zA-Z0–9@:%&._\/+-~?!#=]*)?(\.?\/)?', ' ', string)  # websites
    string = re.sub('\s?http(s)?\:\/+.*|HTTPS?', '', string)  # shortened url
    string = re.sub('w\/|W\/', 'with', string)
    string = re.sub('\\&amp;', ' and ', string)
    string = re.sub('&', ' and ', string)
    string = re.sub('\sPage\s\d{1,2}\:|©(\s*\S*)*', '', string)
    string = re.sub("\@\w+", "@user", string)
    string = re.sub('^,\s?|^\.\s?|^:|\||《|》|•|↝||¤|ï|ð|«|☀|“', '', string)
    string = re.sub("\d{2}\:\d{2}"
                    "|\,?\s?(\d{1,2})?\s?\S{3,10}\s\d{4}\s?(UTC)?"
                    "|\S{3,10}\s\d{1,2}\,\s\d{4}\s?(UTC)?"
                    "|\d{2}\/\d{2}\/\d{2,4}", '', string)
    string = re.sub("\W+\'|\'\W+", "'", string)

    string = html.escape(string)
    string = html.unescape(string)
    string = BeautifulSoup(string, "lxml")
    string = string.get_text()

    new_string = string.split('\"')
    string = "".join(new_string)

    string = re.sub('bb+', 'bb', string)
    string = re.sub('cc+', 'cc', string)
    string = re.sub('dd+', 'dd', string)
    string = re.sub('ff+', 'ff', string)
    string = re.sub('gg+', 'gg', string)
    string = re.sub('hh+', 'hh', string)
    string = re.sub('jj+', 'j', string)
    string = re.sub('kk+', 'kk', string)
    string = re.sub('ll+', 'll', string)
    string = re.sub('mm+', 'mm', string)
    string = re.sub('nn+', 'nn', string)
    string = re.sub('pp+', 'pp', string)
    string = re.sub('qq+', 'q', string)
    string = re.sub('rr+', 'rr', string)
    string = re.sub('ss+', 'ss', string)
    string = re.sub('tt+', 'tt', string)
    string = re.sub('vv+', 'vv', string)
    string = re.sub('ww+', 'ww', string)
    string = re.sub('xx+', 'xx', string)
    string = re.sub('yy+', 'yy', string)
    string = re.sub('zz+', 'zz', string)

    string = re.sub('BB+', 'BB', string)
    string = re.sub('CC+', 'CC', string)
    string = re.sub('DD+', 'DD', string)
    string = re.sub('FF+', 'FF', string)
    string = re.sub('GG+', 'GG', string)
    string = re.sub('HH+', 'HH', string)
    string = re.sub('JJ+', 'JJ', string)
    string = re.sub('KK+', 'KK', string)
    string = re.sub('LL+', 'LL', string)
    string = re.sub('MM+', 'MM', string)
    string = re.sub('NN+', 'NN', string)
    string = re.sub('PP+', 'PP', string)
    string = re.sub('QQ+', 'QQ', string)
    string = re.sub('RR+', 'RR', string)
    string = re.sub('SS+', 'SS', string)
    string = re.sub('TT+', 'TT', string)
    string = re.sub('VV+', 'VV', string)
    string = re.sub('WW+', 'WW', string)
    string = re.sub('XX+', 'XX', string)
    string = re.sub('YY+', 'YY', string)
    string = re.sub('ZZ+', 'ZZ', string)

    string = re.sub('ha(ha)+', 'haha', string)
    string = re.sub('he(he)+', 'hehe', string)
    string = re.sub('hi(hi)+', 'hihi', string)
    string = re.sub('HA(HA)+', 'HAHA', string)
    string = re.sub('HE(HE)+', 'HEHE', string)
    string = re.sub('HI(HI)+', 'HIHI', string)

    string = re.sub('\.\.+', '...', string)
    string = re.sub('\s\.', '.', string)
    string = re.sub('\"{2,}', '"', string)
    string = re.sub('\s\!', '!', string)
    string = re.sub('\s\?', '?', string)
    string = re.sub('\s\,', ',', string)
    string = re.sub('^"+|^\'+|"+$|\'+$', '', string)
    # if it has several types of quotations in the beginning
    string = re.sub('^"+|^\'+|"+$|\'+$', '', string)
    string = re.sub('"+', '"', string)

    # Cyrillic characters [\u4000-\u04ff]
    string = re.sub(r'[а-яА-Я]', '', string)
    string = re.sub(r'[\u4e00-\u9fff]+', '', string)  # Chinese characters
    string = re.sub(r'[\u0621-\u064a\ufb50-\ufdff\ufe70-\ufefc]',
                    '', string)  # Arabic Characters
    string = re.sub('\(\/\S+\)', '', string)
    string = re.sub('\[|\]|\{|\}|\(|\)|\>|\<|\*|\=|\_',
                    '', string)  # e.g., [](){}
    # if it has several types of quotations in the beginning
    string = re.sub('^\"+|^\'+|\"+$|\'+$', '', string)
    string = re.sub('â€™', '\'', string)

    try:
        string = string.encode('latin-1').decode('utf-8')
    except Exception:
        pass

    string = re.sub('^:|^!|^\?|^\-|^\.|^\"|^\/|^\\|$\"', '', string)
    new_string = string.split()
    string = ' '.join(new_string)
    string = re.sub('^:|^!|^\?|^\-|^\.|^\"|^\/|^\\|$\"', '', string)
    new_string = string.split()
    string = ' '.join(new_string)
    new_string = string.split()
    string = ' '.join(new_string)

    if len(string) < 3:
        string = ""

    return string


# %% loading file
df = pd.read_csv('pc_05_train_2013.csv', sep=';')
df_aux_00 = pd.read_csv('pc_05_train_2014.csv', sep=';')
df = pd.concat([df, df_aux_00], axis=0)
del df_aux_00
gc.collect()

df_aux_01 = pd.read_csv('pc_05_train_2015.csv', sep=';')
df = pd.concat([df, df_aux_01], axis=0)
del df_aux_01
gc.collect()

with open('pc_05_train_2016.csv', 'r') as f:
    content = html.unescape(f.read())
df_aux_02 = pd.read_csv(StringIO(content), sep=';')
df = pd.concat([df, df_aux_02], axis=0)
del df_aux_02
gc.collect()

df = df[df["text"] != "unavailable"]
df['text'] = df['text'].apply(lambda x: noise_mitigation(x))
df.assign(
    text=lambda df: df["text"].replace('', np.nan)
).dropna()
df = df[~df["label"].isnull()]
df = df.drop_duplicates(subset=['text'], keep='first')
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df[["text", "label"]]

df_train = df.copy(deep=True)
del df
gc.collect()
# %%

df = pd.read_csv('pc_05_test_2013.csv', sep=';')
df_aux_00 = pd.read_csv('pc_05_test_2014.csv', sep=';')
df = pd.concat([df, df_aux_00], axis=0)
del df_aux_00
gc.collect()

df_aux_01 = pd.read_csv('pc_05_test_2015.csv', sep=';')
df = pd.concat([df, df_aux_01], axis=0)
del df_aux_01
gc.collect()

df_aux_02 = pd.read_csv('pc_05_test_2016.csv', sep=';')
df = pd.concat([df, df_aux_02], axis=0)
del df_aux_02
gc.collect()

df = df[df["text"] != "unavailable"]
df['text'] = df['text'].apply(lambda x: noise_mitigation(x))
df.assign(
    text=lambda df: df["text"].replace('', np.nan)
).dropna()
df = df[~df["label"].isnull()]
df = df.drop_duplicates(subset=['text'], keep='first')
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df[["text", "label"]]

df_test = df.copy(deep=True)

# %%
df_train.to_csv("pc_05_multi_train.csv", sep=";", index=False)
df_test.to_csv("pc_05_multi_test.csv", sep=";", index=False)

# %%
unique_classes = sorted(df_train['label'].unique())

for i in tqdm(range(len(unique_classes))):
    for j in range(i+1, len(unique_classes)):
        # train
        df_aux = df_train.loc[(df_train["label"] == unique_classes[i]) | (
            df_train["label"] == unique_classes[j])]
        df_aux.to_csv(f"pc_05_bin_train_{i}_{j}.csv", sep=";", index=False)

        # test
        df_aux = df_test.loc[(df_test["label"] == unique_classes[i]) | (
            df_test["label"] == unique_classes[j])]
        df_aux.to_csv(f"pc_05_bin_test_{i}_{j}.csv", sep=";", index=False)

# %% saving explained.csv
number_classes = len(unique_classes)
explained_df = pd.DataFrame(
    {
        "index": range(number_classes),
        "label": list(unique_classes)
    }
)
explained_df.to_csv('pc_05_explained.csv', sep=";", index=False)
