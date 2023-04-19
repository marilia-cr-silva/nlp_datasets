# -*- coding: utf-8 -*-
# %% bibtex

'''
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
'''

# %% loading libraries
import pandas as pd
from tqdm import tqdm
import tarfile
import os
import re
import html
import fasttext
from bs4 import BeautifulSoup

# %% loading language detection model
language_identification_model = fasttext.load_model('/mnt/c/Users/Acer/Documents/Corpora/fasttext/lid.176.bin') # or lid.176.ftz lid.176.bin

# %% function to reduce the noise before language identification
def noise_mitigation(aux):
    
    #string = aux.decode('utf-8')
    #string = str(string)
    string = str(aux)
    string = re.sub('\s\#\s|\@user\s?','',string)
    string = re.sub('\-\-+|\s\-\s',' ',string)
    string = re.sub('\s?\@\s',' at ',string)
    string = re.sub(r'\\n','',string)
    string = re.sub(r'(https?:(\/\/))?(www\.)?(\w+)\.(\w+)(\.?\/?)+([a-zA-Z0–9@:%&._\+-?!~#=]*)?(\.?\/)?([a-zA-Z0–9@:%&._\/+-~?!#=]*)?(\.?\/)?',' ',string) # websites
    string = re.sub('w\/|W\/','with',string)
    
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
    string = re.sub('\(|\)|\_','',string)

    new_string = string.split()
    string = ' '.join(new_string)
    
    return string

# %% function to reduce the noise after language identification

def noise_mitigation_lang_id(aux):
    
    string = str(aux)
    string = re.sub(r'[а-яА-Я]','',string) # Cyrillic characters [\u4000-\u04ff]
    string = re.sub(r'[\u4e00-\u9fff]+','',string) # Chinese characters
    string = re.sub(r'[\u0621-\u064a\ufb50-\ufdff\ufe70-\ufefc]','',string) # Arabic Characters
    string = re.sub('\(\/\S+\)','',string) # e.g., (/message/compose/?to=/r/Pikabu)
    string = re.sub('\[|\]|\{|\}|\(|\)|\>|\<|\*|\=','',string) # e.g., [](){}
    new_string = string.split()
    string = ' '.join(new_string)
    
    return string


# %% function to identify language
def detect_language(instance):
    
    aux = str(language_identification_model.predict(instance, k=1)[0][0][-2:])
    
    return aux

# %% loading file
os.chdir('/mnt/c/Users/Acer/Documents/Corpora/downloaded')
tar = tarfile.open("aclImdb_v1.tar.gz", "r:gz")
names = [name for name in tar.getnames()]
names = names[19:50019]

split_label = [[item.split('/')[1],item.split('/')[2]] for item in names]

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

df = pd.concat([df_text,df_labels],axis=1)
df.columns = ['text', 'subset', 'label']
df['text'] = df['text'].apply(lambda x: noise_mitigation(x))
df['language'] = df['text'].apply(lambda x: detect_language(x))
df = df[df['language'] == 'en']
df['text'] = df['text'].apply(lambda x: noise_mitigation_lang_id(x))
df = df.drop_duplicates(subset=['text'],keep='first')
df = df.sample(frac=1,random_state=42).reset_index(drop=True)

# %% splitting train-test
df_test = df[df['subset'] == 'test']
df_test = df_test[['text','label']]
df_train = df[df['subset'] == 'train']
df_train = df_train[['text','label']]

# creating .csv
os.chdir('/mnt/c/Users/Acer/Documents/Corpora/Binary/Sentiment_Analysis/IMDB')
df_train.to_csv(f'SA04_IMDB_Binary_train.csv',sep=';',index=False)
df_test.to_csv(f'SA04_IMDB_Binary_test.csv',sep=';',index=False)