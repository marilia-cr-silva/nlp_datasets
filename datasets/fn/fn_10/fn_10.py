"""
@InProceedings{weinzierl2022vaccinelies,
  author    = {Weinzierl, Maxwell  and  Harabagiu, Sanda},
  title     = {VaccineLies: A Natural Language Resource for Learning to Recognize Misinformation about the COVID-19 and HPV Vaccines},
  booktitle      = {Proceedings of the Language Resources and Evaluation Conference},
  month          = {June},
  year           = {2022},
  address        = {Marseille, France},
  publisher      = {European Language Resources Association},
  pages     = {6967--6975},
  url       = {https://aclanthology.org/2022.lrec-1.753}
}
"""

# %% loading libraries

import pandas as pd
from tqdm import tqdm
import os
import re
import html
from bs4 import BeautifulSoup
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# %% function to reduce the noise

def noise_mitigation(aux):
    
    string = str(aux)
    new_string = string.split('\n')
    string = ' '.join(new_string)
    string = re.sub('\s\#\s|\@user\s?|Says\s|\!+\sRT\s|\s?RT\s|\s?URL','',string)
    string = re.sub('\-\-+|\s\-\s',' ',string)
    string = re.sub('\s?\@\s',' at ',string)
    string = re.sub(r'(https?:(\/\/))?(www\.)?(\w+)\.(\w+)(\.?\/?)+([a-zA-Z0–9@:%&._\+-?!~#=]*)?(\.?\/)?([a-zA-Z0–9@:%&._\/+-~?!#=]*)?(\.?\/)?',' ',string) # websites
    string = re.sub('\s?http(s)?\:\/+.*','',string) # shortened url
    string = re.sub('w\/|W\/','with',string)
    string = re.sub('\\&amp;',' and ',string)
    
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
    string = re.sub('^"+|^\'+|"+$|\'+$','',string)
    string = re.sub('^"+|^\'+|"+$|\'+$','',string) # if it has several types of quotations in the beginning
    string = re.sub('"+','"',string)
    
    string = re.sub(r'[а-яА-Я]','',string) # Cyrillic characters [\u4000-\u04ff]
    string = re.sub(r'[\u4e00-\u9fff]+','',string) # Chinese characters
    string = re.sub(r'[\u0621-\u064a\ufb50-\ufdff\ufe70-\ufefc]','',string) # Arabic Characters
    string = re.sub('\(\/\S+\)','',string) # e.g., (/message/compose/?to=/r/Pikabu)
    string = re.sub('\[|\]|\{|\}|\(|\)|\>|\<|\*|\=|\_','',string) # e.g., [](){}

    new_string = string.split()
    string = ' '.join(new_string)
    
    return string

os.chdir('/mnt/c/Users/Acer/Documents/Corpora/downloaded/FND08')

# %% creating dataframes

df_test_aux = pd.read_csv('FND08_test_with_text.csv',sep=';')
df_validation_aux = pd.read_csv('FND08_dev_with_text.csv',sep=';')
df_test = pd.concat([df_validation_aux,df_test_aux])
df_train = pd.read_csv('FND08_train_with_text.csv',sep=';')

# train set
df_train['text'] = df_train['text'].apply(lambda x: noise_mitigation(x))
df_train.dropna(inplace=True)
df_train = df_train.drop_duplicates(subset=['text'],keep='first')
df_train = df_train.sample(frac=1,random_state=42).reset_index(drop=True)[['text','label']]

# test set
df_test['text'] = df_test['text'].apply(lambda x: noise_mitigation(x))
df_test.dropna(inplace=True)
df_test = df_test.drop_duplicates(subset=['text'],keep='first')
df_test = df_test.sample(frac=1,random_state=42).reset_index(drop=True)[['text','label']]

# %% saving to csv multiclass dataframe
df_train.to_csv(f"fn_10_multi_train.csv",sep=";",index=False)
df_test.to_csv(f"fn_10_multi_test.csv",sep=";",index=False)

# %%
unique_classes = sorted(df_train["label"].unique())

list_csv = []
for i in tqdm(range(len(unique_classes))):
    for j in range(i+1,len(unique_classes)):
        # train
        df_aux = df_train.loc[(df_train["label"] == unique_classes[i]) | (df_train["label"] == unique_classes[j])]
        df_aux.to_csv(f"fn_10_bin_train_{i}_{j}.csv",sep=";",index=False)
        list_csv.append([f"fn_10_bin_train_{i}_{j}.csv",f"{unique_classes[i],unique_classes[j]}"])
        # test
        df_aux = df_test.loc[(df_test["label"] == unique_classes[i]) | (df_test["label"] == unique_classes[j])]
        df_aux.to_csv(f"fn_10_bin_test_{i}_{j}.csv",sep=";",index=False)
        list_csv.append([f"fn_10_bin_test_{i}_{j}.csv",f"{unique_classes[i],unique_classes[j]}"])

df_list_csv = pd.DataFrame(list_csv,columns=["file_name","classes"])
df_list_csv.to_csv("fn_10_bin_explained.csv",sep=";",index=False)

number_classes = unique_classes
explained = {i: unique_classes[i] for i in range(number_classes)}