'''
@inproceedings{gibert2018hate,
    title = "{Hate Speech Dataset from a White Supremacy Forum}",
    author = "de Gibert, Ona  and
      Perez, Naiara  and
      Garc{'\i}a-Pablos, Aitor  and
      Cuadros, Montse",
    booktitle = "Proceedings of the 2nd Workshop on Abusive Language Online ({ALW}2)",
    month = oct,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W18-5102",
    doi = "10.18653/v1/W18-5102",
    pages = "11--20",
}
'''
# %%
# %% loading libraries
import pandas as pd
import numpy as np
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
    string = re.sub(r'\n|\t|\\n|\\t','',string)
    string = re.sub('\s\#\s|\@user\s?|\s\#\s|\@USER\s?|Says\s|\!+\sRT\s|\s?RT\s','',string)
    string = re.sub('\-\-+|\s\-\s',' ',string)
    string = re.sub('\s?\@\s',' at ',string)
    string = re.sub(r'(https?:(\/\/))?(www\.)?(\w+)\.(\w+)(\.?\/?)+([a-zA-Z0–9@:%&._\+-?!~#=]*)?(\.?\/)?([a-zA-Z0–9@:%&._\/+-~?!#=]*)?(\.?\/)?',' ',string) # websites
    string = re.sub('\s?http(s)?\:\/+.*|HTTPS?','',string) # shortened url
    string = re.sub('w\/|W\/','with',string)
    string = re.sub('\\&amp;',' and ',string)
    string = re.sub('&',' and ',string)
    string = re.sub('\sPage\s\d{1,2}\:|©(\s*\S*)*','',string)
    string = re.sub("\@\w+","@user",string)
    string = re.sub('^,\s?|^\.\s?|^:|\||《|》|•','',string)
    string = re.sub("\d{2}\:\d{2}"
                    "|\,?\s?(\d{1,2})?\s?\S{3,10}\s\d{4}\s?(UTC)?"
                    "|\S{3,10}\s\d{1,2}\,\s\d{4}\s?(UTC)?"
                    "|\d{2}\/\d{2}\/\d{2,4}",'',string)
    string = re.sub("\W+\'|\'\W+","'",string)

    string = html.escape(string)
    string = html.unescape(string)
    string = BeautifulSoup(string, "lxml")
    string = string.get_text()

    new_string = string.split('\"')
    string = "".join(new_string)

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
    string = re.sub('qq+','qq',string)
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
    string = re.sub('^\"+|^\'+|\"+$|\'+$','',string) # if it has several types of quotations in the beginning
    string = re.sub('â€™','\'',string)
    string = re.sub('^:|^!|^\?|^\-|^\.|^\"|^\/|^\\|$\"','',string)
    new_string = string.split()
    string = ' '.join(new_string)

    if len(string) < 3:
        string = ""
    
    return string
# %%
dataset_name = 'hs_03'

# %%
def goto_root():
    while os.path.basename(os.getcwd()).upper() != dataset_name.upper():
        os.chdir('../')

# %%
annotations_df = pd.read_csv('hs_03_metadata.csv')
annotations_df['text'] = [''] * annotations_df.shape[0]
annotations_df['partition'] = [0] * annotations_df.shape[0]

# %% Get test files
goto_root()
os.chdir('./hs_03_sampled_test/')
test_files = os.listdir()

# %% Get train files
goto_root()
os.chdir('./hs_03_sampled_train/')
train_files = os.listdir()


# %%
goto_root()
for test_file in test_files:
    with open(f'hs_03_sampled_test/{test_file}') as f:
        content = f.read()
    file_id = test_file.split('.txt')[0]
    idx = annotations_df.index[annotations_df['file_id'] == file_id]
    annotations_df.loc[idx, 'text'] = content
    annotations_df.loc[idx, 'partition'] = 1

# %%
goto_root()
for train_file in train_files:
    with open(f'hs_03_sampled_train/{train_file}', encoding='utf-8') as f:
        content = f.read()
        
    file_id = train_file.split('.txt')[0]
    idx = annotations_df.index[annotations_df['file_id'] == file_id]
    annotations_df.loc[idx, 'text'] = content
    annotations_df.loc[idx, 'partition'] = 2 

# %%
df_test = annotations_df[annotations_df['partition'] == 1]
df_test = df_test[["text", "label"]]
df_test['text'] = df_test['text'].apply(lambda x: noise_mitigation(x))
df_test = df_test.drop_duplicates(subset=['text'],keep='first')
df_test.assign(
        text=lambda df : df["text"].replace('', np.nan)
    ).dropna().reset_index(drop=True)
df_test = df_test.sample(frac=1,random_state=42).reset_index(drop=True)


# %%
df_train = annotations_df[annotations_df['partition'] == 2]
df_train = df_train[["text", "label"]]
df_train['text'] = df_train['text'].apply(lambda x: noise_mitigation(x))
df_train = df_train.drop_duplicates(subset=['text'],keep='first')
df_train.assign(
        text=lambda df : df["text"].replace('', np.nan)
    ).dropna().reset_index(drop=True)
df_train = df_train.sample(frac=1,random_state=42).reset_index(drop=True)

# %%
goto_root()
df_train.to_csv('hs_03_bin_train_0_1.csv', sep=';', index=False)
df_test.to_csv('hs_03_bin_test_0_1.csv', sep=';', index=False)

# %% saving explained.csv
unique_classes = sorted(df_train["label"].unique())

number_classes = len(unique_classes)
explained_df = pd.DataFrame(
    {
        "index": range(number_classes),
        "label": list(unique_classes)
    }
)
explained_df.to_csv('hs_03_explained.csv', sep = ";", index=False)
