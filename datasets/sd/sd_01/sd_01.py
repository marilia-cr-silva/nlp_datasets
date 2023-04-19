# %% bibtex

'''
"@article{kiesel2019datahyperpartisansemeval,
  title={Data for pan at semeval 2019 task 4: Hyperpartisan news detection},
  author={Kiesel, Johannes and Mestre, Maria and Shukla, Rishabh and Vincent, Emmanuel and Corney, David and Adineh, Payam and Stein, Benno and Potthast, Martin},
  year={2019}
}"
'''

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

def noise_mitigation(aux):
    
    string = str(aux)
    new_string = string.split('\n')
    string = ' '.join(new_string)
    string = re.sub('\s\#\s|\@user\s?','',string)
    string = re.sub('\-\-+|\s\-\s',' ',string)
    string = re.sub('\s?\@\s',' at ',string)
    string = re.sub(r'(https?:(\/\/))?(www\.)?(\w+)\.(\w+)(\.?\/?)+([a-zA-Z0–9@:%&._\+-?!~#=]*)?(\.?\/)?([a-zA-Z0–9@:%&._\/+-~?!#=]*)?(\.?\/)?',' ',string) # websites
    string = re.sub('w\/|W\/','with',string)
    string = re.sub('\\&amp;',' and ',string)
    string = re.sub('\sPage\s\d{1,2}\:|©(\s*\S*)*','',string) # e.g., © 2018 Cox Media Group. Page 16: © 2017 Cable News Network
    
    string = html.escape(string)
    string = html.unescape(string)
    string = BeautifulSoup(string, "lxml")
    string = string.get_text()

    string = re.sub('aa+','aa',string)
    string = re.sub('bb+','bb',string)
    string = re.sub('cc+','cc',string)
    string = re.sub('dd+','dd',string)
    string = re.sub('ee+','ee',string)
    string = re.sub('ff+','ff',string)
    string = re.sub('gg+','gg',string)
    string = re.sub('hh+','hh',string)
    string = re.sub('ii+','ii',string)
    string = re.sub('jj+','j',string)
    string = re.sub('kk+','kk',string)
    string = re.sub('ll+','ll',string)
    string = re.sub('mm+','mm',string)
    string = re.sub('nn+','nn',string)
    string = re.sub('oo+','oo',string)
    string = re.sub('pp+','pp',string)
    string = re.sub('qq+','q',string)
    string = re.sub('rr+','rr',string)
    string = re.sub('ss+','ss',string)
    string = re.sub('tt+','tt',string)
    string = re.sub('uu+','uu',string)
    string = re.sub('vv+','vv',string)
    string = re.sub('ww+','ww',string)
    string = re.sub('xx+','xx',string)
    string = re.sub('yy+','yy',string)
    string = re.sub('zz+','zz',string)

    string = re.sub('AA+','AA',string)
    string = re.sub('BB+','BB',string)
    string = re.sub('CC+','CC',string)
    string = re.sub('DD+','DD',string)
    string = re.sub('EE+','EE',string)
    string = re.sub('FF+','FF',string)
    string = re.sub('GG+','GG',string)
    string = re.sub('HH+','HH',string)
    string = re.sub('II+','II',string)
    string = re.sub('JJ+','JJ',string)
    string = re.sub('KK+','KK',string)
    string = re.sub('LL+','LL',string)
    string = re.sub('MM+','MM',string)
    string = re.sub('NN+','NN',string)
    string = re.sub('OO+','OO',string)
    string = re.sub('PP+','PP',string)
    string = re.sub('QQ+','q',string)
    string = re.sub('RR+','RR',string)
    string = re.sub('SS+','SS',string)
    string = re.sub('TT+','TT',string)
    string = re.sub('UU+','UU',string)
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
    string = re.sub('(\!+\?+)+','?!',string)
    string = re.sub('\!\!+','!',string)
    string = re.sub('\s\!','!',string)
    string = re.sub('\?\?+','?',string)
    string = re.sub('\s\?','?',string)
    string = re.sub('\s\,',',',string)
    
    string = re.sub(r'[а-яА-Я]','',string) # Cyrillic characters [\u4000-\u04ff]
    string = re.sub(r'[\u4e00-\u9fff]+','',string) # Chinese characters
    string = re.sub(r'[\u0621-\u064a\ufb50-\ufdff\ufe70-\ufefc]','',string) # Arabic Characters
    string = re.sub('\(\/\S+\)','',string) # e.g., (/message/compose/?to=/r/Pikabu)
    string = re.sub('\[|\]|\{|\}|\(|\)|\>|\<|\*|\=|\_','',string) # e.g., [](){}

    new_string = string.split()
    string = ' '.join(new_string)
    
    return string

# loading datasets

dataset = load_dataset("hyperpartisan_news_detection",'byarticle') # by article

# %% creating dataframes

df = pd.DataFrame([dataset['train']['text'],dataset['train']['hyperpartisan']]).T
df.columns = ['text','label']

# train set
df['text'] = df['text'].apply(lambda x: noise_mitigation(x))
df = df.drop_duplicates(subset=['text'],keep='first')
df = df.sample(frac=1,random_state=42).reset_index(drop=True)

# %%
df_train, df_test = train_test_split(
    df, test_size=0.3, random_state=42,shuffle = True)

df_train.to_csv(f"sd_01_bin_train.csv",sep=";",index=False)
df_test.to_csv(f"sd_01_bin_test.csv",sep=";",index=False)

# %%
unique_classes = sorted(df_train["label"].unique())

number_classes = unique_classes
explained = {i: unique_classes[i] for i in range(number_classes)}
