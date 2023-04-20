'''
@inproceedings{basile-etal-2019-semeval-hateval,
    title = "{S}em{E}val-2019 Task 5: Multilingual Detection of Hate Speech Against Immigrants and Women in Twitter",
    author = "Basile, Valerio  and
      Bosco, Cristina  and
      Fersini, Elisabetta  and
      Nozza, Debora  and
      Patti, Viviana  and
      Rangel Pardo, Francisco Manuel  and
      Rosso, Paolo  and
      Sanguinetti, Manuela",
    booktitle = "Proceedings of the 13th International Workshop on Semantic Evaluation",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/S19-2007",
    doi = "10.18653/v1/S19-2007",
    pages = "54--63",
}
'''

# %% loading libraries
import pandas as pd
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
    string = re.sub('\s\#\s|\@user\s?|Says\s|\!+\sRT\s|\s?RT\s|\s?URL','',string)
    string = re.sub('\-\-+|\s\-\s',' ',string)
    string = re.sub('\s?\@\s',' at ',string)
    string = re.sub(r'(https?:(\/\/))?(www\.)?(\w+)\.(\w+)(\.?\/?)+([a-zA-Z0–9@:%&._\+-?!~#=]*)?(\.?\/)?([a-zA-Z0–9@:%&._\/+-~?!#=]*)?(\.?\/)?',' ',string) # websites
    string = re.sub('\s?http(s)?\:\/+.*','',string) # shortened url
    string = re.sub('w\/|W\/','with',string)
    string = re.sub('\\&amp;',' and ',string)
#    string = re.sub('\@\S*\s','',string) # it removes mentions
    
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
    try:
        string = string.encode('latin-1').decode('utf-8')
    except:
        pass

    new_string = string.split()
    string = ' '.join(new_string)
    
    return string

# %% changing file directory
os.chdir('/mnt/c/Users/Acer/Documents/Corpora/downloaded/HS06')
list_files = os.listdir()

# %%
os.chdir('/mnt/c/Users/Acer/Documents/Corpora/downloaded/HS06')

# %% creating dataframes

df_test_aux = pd.read_csv(list_files[1])
df_validation_aux = pd.read_csv(list_files[0])
df_test = pd.concat([df_validation_aux,df_test_aux])
df_train = pd.read_csv(list_files[2])

# %%
df_train = df_train[['text','HS']] # only general hate speech (without specified target)
df_train.columns = ['text','label']
df_test = df_test[['text','HS']] # only general hate speech (without specified target)
df_test.columns = ['text','label']

# %% train set
df_train['text'] = df_train['text'].apply(lambda x: noise_mitigation(x))
df_train.dropna(inplace=True)
df_train = df_train.drop_duplicates(subset=['text'],keep='first')
df_train = df_train.sample(frac=1,random_state=42).reset_index(drop=True)[['text','label']]

# %% test set
df_test['text'] = df_test['text'].apply(lambda x: noise_mitigation(x))
df_test.dropna(inplace=True)
df_test = df_test.drop_duplicates(subset=['text'],keep='first')
df_test = df_test.sample(frac=1,random_state=42).reset_index(drop=True)[['text','label']]

# %% saving to csv multiclass dataframe
os.chdir('/mnt/c/Users/Acer/Documents/Corpora/Binary/Hate_Speech_Detection')
df_train.to_csv('HS06_Binary_train.csv',sep=';',index=False)
df_test.to_csv('HS06_Binary_test.csv',sep=';',index=False)