# -*- coding: utf-8 -*-
# %% bibtex

'''
@inproceedings{10.1145/2507157.2507163_amazon_polarity,
    author = {McAuley, Julian and Leskovec, Jure},
    title = {Hidden Factors and Hidden Topics: Understanding Rating Dimensions with Review Text},
    year = {2013},
    isbn = {9781450324090},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/2507157.2507163},
    doi = {10.1145/2507157.2507163},
    abstract = {In order to recommend products to users we must ultimately predict how a user will respond to a new product. To do so we must uncover the implicit tastes of each user as well as the properties of each product. For example, in order to predict whether a user will enjoy Harry Potter, it helps to identify that the book is about wizards, as well as the user's level of interest in wizardry. User feedback is required to discover these latent product and user dimensions. Such feedback often comes in the form of a numeric rating accompanied by review text. However, traditional methods often discard review text, which makes user and product latent dimensions difficult to interpret, since they ignore the very text that justifies a user's rating. In this paper, we aim to combine latent rating dimensions (such as those of latent-factor recommender systems) with latent review topics (such as those learned by topic models like LDA). Our approach has several advantages. Firstly, we obtain highly interpretable textual labels for latent rating dimensions, which helps us to `justify' ratings with text. Secondly, our approach more accurately predicts product ratings by harnessing the information present in review text; this is especially true for new products and users, who may have too few ratings to model their latent factors, yet may still provide substantial information from the text of even a single review. Thirdly, our discovered topics can be used to facilitate other tasks such as automated genre discovery, and to identify useful and representative reviews.},
    booktitle = {Proceedings of the 7th ACM Conference on Recommender Systems},
    pages = {165–172},
    numpages = {8},
    keywords = {recommender systems, topic models},
    location = {Hong Kong, China},
    series = {RecSys '13}
}
'''

import html
import os
import re

import fasttext
import numpy as np
# %% loading libraries
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from datasets import load_dataset

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


def detect_language(instance):

    aux = str(language_identification_model.predict(instance, k=1)[0][0][-2:])

    return aux


# %% loading dataset
dataset = load_dataset("amazon_polarity")

# %% creating dataframes
df_test = pd.DataFrame([dataset['test']['title'],
                       dataset['test']['content'], dataset['test']['label']]).T
df_test.columns = ['title', 'content', 'label']
df_test['text'] = df_test['title'] + ' ' + df_test['content']
df_test = df_test[['text', 'label']]
df_train = pd.DataFrame([dataset['train']['title'],
                        dataset['train']['content'], dataset['train']['label']]).T
df_train.columns = ['title', 'content', 'label']
df_train['text'] = df_train['title'] + ' ' + df_train['content']
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
df_train.to_csv('pc_04_bin_train_0_1.csv', sep=';', index=False)
df_test.to_csv('pc_04_bin_test_0_1.csv', sep=';', index=False)

# %% saving explained.csv
unique_classes = sorted(df_train["label"].unique())

number_classes = len(unique_classes)
explained_df = pd.DataFrame(
    {
        "index": range(number_classes),
        "label": list(unique_classes)
    }
)
explained_df.to_csv('pc_04_explained.csv', sep=";", index=False)
