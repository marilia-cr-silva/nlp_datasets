"""
@inproceedings{Perez_Rosas18Automatic_fakenewsamt,
    author = {Verónica Pérez-Rosas and Bennett Kleinberg and Alexandra Lefevre and Rada Mihalcea},
    title = {\href{https://aclanthology.org/C18-1287/}{Automatic Detection of Fake News}},
    publisher = "Association for Computational Linguistics",
    journal = {COLING},
    address = "Santa Fe, New Mexico, USA",
    year = {2018}
}
https://lit.eecs.umich.edu/downloads.html#Fake%20News
http://web.eecs.umich.edu/~mihalcea/downloads/fakeNewsDatasets.zip
"""

# %%
# %% loading libraries

import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import os
import re
import html
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore")

# %%
dataset_name = "fn_02"

# %% function to reduce the noise
def noise_mitigation(aux):
    
    string = str(aux)
    new_string = string.split('\n')
    string = ' '.join(new_string)
    string = re.sub(r'\n|\t|\\n|\\t','',string)
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

# %%
def goto_root():
    while os.path.basename(os.getcwd()).upper() != dataset_name.upper():
        os.chdir('../')

# %% reading fake files
goto_root()
os.chdir('./fakeNewsDataset')
os.chdir('./fake')
fake_files = os.listdir()

# %% reading legit files
goto_root()
os.chdir('./fakeNewsDataset')
os.chdir('./legit')
legit_files = os.listdir()

# %% reading fake files content

goto_root()
os.chdir('./fakeNewsDataset/fake/')
fake_content = []
for f in fake_files:
    with open(f, 'r') as f:
        fake_content.append(f.read())


# %% reading legit files

goto_root()
os.chdir('./fakeNewsDataset/legit/')
legit_content = []
for f in legit_files:
    with open(f, 'r') as f:
        legit_content.append(f.read())


# %% creating dataframes

df_fake = pd.DataFrame(
    data={'text': fake_content, 'label': ['fake'] * len(fake_content)})
df_legit = pd.DataFrame(
    data={'text': legit_content, 'label': ['legit'] * len(legit_content)})

df_fake_news = df_fake.append(df_legit, ignore_index=True)

# %% saving csv
goto_root()
df_fake_news = df_fake_news.drop_duplicates(subset=["text"],keep="first")
df_train, df_test = train_test_split(
    df_fake_news, test_size=0.3, random_state=42,shuffle = True)

df_train["new_text"] = df_train["text"].apply(lambda x: noise_mitigation(x))
df_train.assign(
        text=lambda df : df["text"].replace('', np.nan)
    ).dropna().reset_index(drop=True)
df_train = df_train.drop_duplicates(subset=["new_text"],keep="first")
df_train = df_train[["new_text","label"]]
df_train.rename(columns={"new_text": "text"}, inplace=True)
df_train = df_train.sample(frac=1,random_state=42).reset_index(drop=True)

df_test["new_text"] = df_test["text"].apply(lambda x: noise_mitigation(x))
df_test.assign(
        text=lambda df : df["text"].replace('', np.nan)
    ).dropna().reset_index(drop=True)
df_test = df_test.drop_duplicates(subset=["new_text"],keep="first")
df_test = df_test[["new_text","label"]]
df_test.rename(columns={"new_text": "text"}, inplace=True)
df_test = df_test.sample(frac=1,random_state=42).reset_index(drop=True)

# %%


df_train.to_csv('fn_02_bin_train_0_1.csv', sep=';', index=False)
df_test.to_csv('fn_02_bin_test_0_1.csv', sep=';', index=False)

# %% saving explained.csv
unique_classes = sorted(df_fake_news["label"].unique())

number_classes = len(unique_classes)
explained_df = pd.DataFrame(
    {
        "index": range(number_classes),
        "label": list(unique_classes)
    }
)
explained_df.to_csv('fn_02_explained.csv', sep = ";", index=False)
