# -*- coding: utf-8 -*-
# %% bibtex

'''
@InProceedings{waseem-hovy:2016:N16-2_hate_annotation,
  author    = {Waseem, Zeerak  and  Hovy, Dirk},
  title     = {Hateful Symbols or Hateful People? Predictive Features for Hate Speech Detection on Twitter},
  booktitle = {Proceedings of the NAACL Student Research Workshop},
  month     = {June},
  year      = {2016},
  address   = {San Diego, California},
  publisher = {Association for Computational Linguistics},
  pages     = {88--93},
  url       = {http://www.aclweb.org/anthology/N16-2013}
}
'''

# %% loading libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import re
import html
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# %% function to reduce the noise

def noise_mitigation(aux):
    
    string = str(aux)
    new_string = string.split('\n')
    string = ' '.join(new_string)
    string = re.sub('\s\#\s|\@user\s?|\s\#\s|\@USER\s?|Says\s|\!+\sRT\s|\s?RT\s','',string)
    string = re.sub('\-\-+|\s\-\s',' ',string)
    string = re.sub('\s?\@\s',' at ',string)
    string = re.sub(r'(https?:(\/\/))?(www\.)?(\w+)\.(\w+)(\.?\/?)+([a-zA-Z0–9@:%&._\+-?!~#=]*)?(\.?\/)?([a-zA-Z0–9@:%&._\/+-~?!#=]*)?(\.?\/)?',' ',string) # websites
    string = re.sub('\s?http(s)?\:\/+.*|HTTPS?','',string) # shortened url
    string = re.sub('w\/|W\/','with',string)
    string = re.sub('\\&amp;',' and ',string)
    string = re.sub('\sPage\s\d{1,2}\:|©(\s*\S*)*','',string)
    string = re.sub("\@\w+","@user",string)
    
    string = html.escape(string)
    string = html.unescape(string)
    string = BeautifulSoup(string, "lxml")
    string = string.get_text()

    string = re.sub('bb+','bb',string)
    string = re.sub('cc+','cc',string)
    string = re.sub('dd+','dd',string)
    string = re.sub('ff+','ff',string)
    string = re.sub('gg+','gg',string)
    string = re.sub('hh+','hh',string)
    string = re.sub('jj+','j',string)
    string = re.sub('kk+','kk',string)
    string = re.sub('ll+','ll',string)
    string = re.sub('mm+','mm',string)
    string = re.sub('nn+','nn',string)
    string = re.sub('pp+','pp',string)
    string = re.sub('qq+','q',string)
    string = re.sub('rr+','rr',string)
    string = re.sub('ss+','ss',string)
    string = re.sub('tt+','tt',string)
    string = re.sub('vv+','vv',string)
    string = re.sub('ww+','ww',string)
    string = re.sub('xx+','xx',string)
    string = re.sub('yy+','yy',string)
    string = re.sub('zz+','zz',string)

    string = re.sub('BB+','BB',string)
    string = re.sub('CC+','CC',string)
    string = re.sub('DD+','DD',string)
    string = re.sub('FF+','FF',string)
    string = re.sub('GG+','GG',string)
    string = re.sub('HH+','HH',string)
    string = re.sub('JJ+','JJ',string)
    string = re.sub('KK+','KK',string)
    string = re.sub('LL+','LL',string)
    string = re.sub('MM+','MM',string)
    string = re.sub('NN+','NN',string)
    string = re.sub('PP+','PP',string)
    string = re.sub('QQ+','QQ',string)
    string = re.sub('RR+','RR',string)
    string = re.sub('SS+','SS',string)
    string = re.sub('TT+','TT',string)
    string = re.sub('VV+','VV',string)
    string = re.sub('WW+','WW',string)
    string = re.sub('XX+','XX',string)
    string = re.sub('YY+','YY',string)
    string = re.sub('ZZ+','ZZ',string)

    string = re.sub('ha(ha)+','haha',string)
    string = re.sub('he(he)+','hehe',string)
    string = re.sub('hi(hi)+','hihi',string)
    string = re.sub('HA(HA)+','HAHA',string)
    string = re.sub('HE(HE)+','HEHE',string)
    string = re.sub('HI(HI)+','HIHI',string)

    string = re.sub('\.\.+','...',string)
    string = re.sub('\s\.','.',string)
    string = re.sub('\"{2,}','"',string)
    string = re.sub('\s\!','!',string)
    string = re.sub('\s\?','?',string)
    string = re.sub('\s\,',',',string)
    string = re.sub('^"+|^\'+|"+$|\'+$','',string)
    string = re.sub('^"+|^\'+|"+$|\'+$','',string) # if it has several types of quotations in the beginning
    string = re.sub('"+','"',string)
    
    string = re.sub(r'[а-яА-Я]','',string) # Cyrillic characters [\u4000-\u04ff]
    string = re.sub(r'[\u4e00-\u9fff]+','',string) # Chinese characters
    string = re.sub(r'[\u0621-\u064a\ufb50-\ufdff\ufe70-\ufefc]','',string) # Arabic Characters
    string = re.sub('\(\/\S+\)','',string)
    string = re.sub('\[|\]|\{|\}|\(|\)|\>|\<|\*|\=|\_','',string) # e.g., [](){}

    new_string = string.split()
    string = ' '.join(new_string)

    if len(string) < 3:
        string = ""
    
    return string

# %% loading dataset
df = pd.read_csv('hs_01.csv',sep=';')

# %% excluding instances with 'racism' label (only 12) and the ones without text data ('banned_user')
df = df[df['text'] != 'banned_user']
df = df[df['label'] != 'racism']
df = df[['text','label','id']]

# %% preprocessing
df['text'] = df['text'].apply(lambda x: noise_mitigation(x))
df = df.drop_duplicates(subset=['text'],keep='first')
df.loc[df["label"].isnull(), ["label"]] = "none"

df_train, df_test = train_test_split(
    df, test_size=0.3, random_state=42,shuffle = True)

df_train.assign(
        text=lambda df : df["text"].replace('', np.nan)
    ).dropna().reset_index(drop=True)
df_train = df_train.sample(frac=1,random_state=42).reset_index(drop=True)

df_test.assign(
        text=lambda df : df["text"].replace('', np.nan)
    ).dropna().reset_index(drop=True)
df_test = df_test.sample(frac=1,random_state=42).reset_index(drop=True)

# %%
df_train.to_csv('hs_01_bin_train_0_1.csv', sep=';', index=False)
df_test.to_csv('hs_01_bin_test_0_1.csv', sep=';', index=False)

# %% saving explained.csv
unique_classes = sorted(df["label"].unique())

number_classes = len(unique_classes)
explained_df = pd.DataFrame(
    {
        "index": range(number_classes),
        "label": list(unique_classes)
    }
)
explained_df.to_csv('hs_01_explained.csv', sep = ";", index=False)

# %%
