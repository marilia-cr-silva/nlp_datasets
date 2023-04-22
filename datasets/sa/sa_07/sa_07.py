'''
@inproceedings{NIPS2015_250cf8b5_ag_news_yelp,
 author = {Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {C. Cortes and N. Lawrence and D. Lee and M. Sugiyama and R. Garnett},
 pages = {},
 publisher = {Curran Associates, Inc.},
 title = {Character-level Convolutional Networks for Text Classification},
 url = {https://proceedings.neurips.cc/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf},
 volume = {28},
 year = {2015}
}

https://www.tensorflow.org/datasets/catalog/yelp_polarity_reviews?hl=en
https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz
'''

# %% loading libraries
import pandas as pd
from tqdm import tqdm
#from datasets import load_dataset
import os
import re
import html
import fasttext
from bs4 import BeautifulSoup
#from zipfile import ZipFile
import tarfile
import csv
import io

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

# %% changing directory
os.chdir('/mnt/c/Users/Acer/Documents/Corpora/downloaded')

# % opening file
with tarfile.open('yelp_review_polarity_csv.tgz') as tar:
    
    name = [name for name in tar.getnames()]
    name = name[1] + ' ' + name[3]
    list_files = name.split()
    list_rows = []
    
    for i in range(len(list_files)):
        csv_file = io.StringIO(tar.extractfile(list_files[i]).read().decode('utf8'))
    
        for row in csv.reader(csv_file):
            list_rows.append(row)
        
        # %% creating dataframes
        if i == 0:
            df_train = pd.DataFrame(list_rows)
            df_train.columns = ['label', 'text']
            df_train = df_train[['text','label']]
        else:
            df_test = pd.DataFrame(list_rows)
            df_test.columns = ['label', 'text']
            df_test = df_test[['text','label']]            

# %% train set
df_train['text'] = df_train['text'].apply(lambda x: noise_mitigation(x))
df_train['language'] = df_train['text'].apply(lambda x: detect_language(x))
df_train = df_train[df_train['language'] == 'en']
df_train['text'] = df_train['text'].apply(lambda x: noise_mitigation_lang_id(x))
df_train = df_train.drop_duplicates(subset=['text'],keep='first')
df_train = df_train.sample(frac=1,random_state=42).reset_index(drop=True)[['text','label']]


# %% test set
df_test['text'] = df_test['text'].apply(lambda x: noise_mitigation(x))
df_test['language'] = df_test['text'].apply(lambda x: detect_language(x))
df_test = df_test[df_test['language'] == 'en']
df_test['text'] = df_test['text'].apply(lambda x: noise_mitigation_lang_id(x))
df_test = df_test.drop_duplicates(subset=['text'],keep='first')
df_test = df_test.sample(frac=1,random_state=42).reset_index(drop=True)[['text','label']]

# %% creating .csv
os.chdir('/mnt/c/Users/Acer/Documents/Corpora/Binary/Sentiment_Analysis/Yelp')
df_train.to_csv(f'SA07_Yelp_Binary_train.csv',sep=';',index=False)
df_test.to_csv(f'SA07_Yelp_Binary_test.csv',sep=';',index=False)