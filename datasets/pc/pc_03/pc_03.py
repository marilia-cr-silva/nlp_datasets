# -*- coding: utf-8 -*-
# %% bibtex

"""
@inproceedings{maas-etal-2011-learning-imdb,
    title = "Learning Word Vectors for Sentiment Analysis",
    author = "Maas, Andrew L.  and
      Daly, Raymond E.  and
      Pham, Peter T.  and
      Huang, Dan  and
      Ng, Andrew Y.  and
      Potts, Christopher",
    booktitle = "Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2011",
    address = "Portland, Oregon, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P11-1015",
    pages = "142--150",

download
http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
}
"""
# %% loading libraries
import html
import os
import re
import tarfile
import fasttext
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# %% loading language detection model
language_identification_model = fasttext.load_model(
    'lid.176.bin')  # or lid.176.ftz lid.176.bin

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

# %% function to identify language


def detect_language(instance) -> str:
    aux = str(language_identification_model.predict(instance, k=1)[0][0][-2:])

    return aux


# %% loading file
tar = tarfile.open("aclImdb_v1.tar.gz", "r:gz")

# %%
reg_exp = "^aclImdb/(train|test)/(neg|pos)/\w+\.txt$"
names = [str(name) for name in tar.getnames()]
names = [name for name in names if re.search(reg_exp, name) != None]

# %%
split_label = [[item.split('/')[1], item.split('/')[2]] for item in names]

# %%
list_comments = []
for element in tqdm(names):
    f = tar.extractfile(element)
    if f is not None:
        aux = f.read()
        aux_dec = aux.decode('utf-8')
        list_comments.append(aux_dec)

# %% creating dataframes
df_text = pd.DataFrame(list_comments)
df_labels = pd.DataFrame(split_label)

df = pd.concat([df_text, df_labels], axis=1)
df.columns = ['text', 'subset', 'label']

# %% splitting train-test
df_test = df[df['subset'] == 'test']
df_test = df_test[['text', 'label']]
df_train = df[df['subset'] == 'train']
df_train = df_train[['text', 'label']]

# %% train set
df_train['text'] = df_train['text'].apply(lambda x: noise_mitigation(x))
df_train['language'] = df_train['text'].apply(lambda x: detect_language(x))
df_train = df_train[df_train['language'] == 'en']
df_train = df_train[['text', 'label']]
df_train.assign(
    text=lambda df: df["text"].replace('', np.nan)
).dropna().reset_index(drop=True)
df_train = df_train.drop_duplicates(subset=['text'], keep='first')
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

# %% test set
df_test['text'] = df_test['text'].apply(lambda x: noise_mitigation(x))
df_test['language'] = df_test['text'].apply(lambda x: detect_language(x))
df_test = df_test[df_test['language'] == 'en']
df_test = df_test[['text', 'label']]
df_test.assign(
    text=lambda df: df["text"].replace('', np.nan)
).dropna().reset_index(drop=True)
df_test = df_test.drop_duplicates(subset=['text'], keep='first')
df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)

# %%
df_train.to_csv('pc_03_bin_train_0_1.csv', sep=';', index=False)
df_test.to_csv('pc_03_bin_test_0_1.csv', sep=';', index=False)

# %% saving explained.csv
unique_classes = sorted(df_train["label"].unique())

number_classes = len(unique_classes)
explained_df = pd.DataFrame(
    {
        "index": range(number_classes),
        "label": list(unique_classes)
    }
)
explained_df.to_csv('pc_03_explained.csv', sep=";", index=False)
