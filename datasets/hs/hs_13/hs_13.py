
'''
@article{mathew2020hatexplain,
      title={HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection},
      author={Binny Mathew and Punyajoy Saha and Seid Muhie Yimam and Chris Biemann and Pawan Goyal and Animesh Mukherjee},
      year={2021},
      conference={AAAI conference on artificial intelligence}
}
'''

# %% loading libraries
import pandas as pd
from tqdm import tqdm
import os
import re
import html
import gc
from bs4 import BeautifulSoup
from collections import Counter
from datasets import load_dataset
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
    string = re.sub('\\&amp;|\s\&\s',' and ',string)
    string = re.sub('^,\s?|^\.\s?|^:|\||《|》|•','',string)
    string = re.sub("\d{2}\:\d{2}"
                    "|\,?\s?(\d{1,2})?\s?\S{3,10}\s\d{4}\s?(UTC)?"
                    "|\S{3,10}\s\d{1,2}\,\s\d{4}\s?(UTC)?"
                    "|\d{2}\/\d{2}\/\d{2,4}",'',string) # time, dates
    #    string = re.sub('\@\S*\s','',string)

    string = html.escape(string)
    string = html.unescape(string)
    string = BeautifulSoup(string, "lxml")
    string = string.get_text()


    new_string = string.split('\"')
    string = "".join(new_string)    

    new_string = string.split('\'')
    string = "".join(new_string) 

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
    string = re.sub('\:{2,}',':',string)
    string = re.sub('\"{2,}','"',string)
    string = re.sub('(\!+\?+)+','?!',string)
    string = re.sub('\!\!+','!',string)
    string = re.sub('\s\!','!',string)
    string = re.sub('\?\?+','?',string)
    string = re.sub('\s\?','?',string)
    string = re.sub('\s\,',',',string)
    #    string = re.sub('^\"+|^\'+|\"+$|\'+$','',string)
    #    string = re.sub('^\"+|^\'+|\"+$|\'+$','',string) # if it has several types of quotations in the beginning
    
    string = re.sub(r'[а-яА-Я]','',string) # Cyrillic characters [\u4000-\u04ff]
    string = re.sub(r'[\u4e00-\u9fff]+','',string) # Chinese characters
    string = re.sub(r'[\u0621-\u064a\ufb50-\ufdff\ufe70-\ufefc]','',string) # Arabic Characters
    string = re.sub('\(\/\S+\)','',string) # e.g., (/message/compose/?to=/r/Pikabu)
    string = re.sub('\[|\]|\{|\}|\(|\)|\>|\<|\*|\=|\_|\+','',string) # e.g., [](){}
    string = re.sub('â€™','\'',string)
    string = re.sub('^:|^!|^\?|^\-|^\.|^\"|^\/|^\\|$\"','',string)
    
    try:
        string = string.encode('latin-1').decode('utf-8')
    except:
        pass
    
    string = re.sub('^:|^!|^\?|^\-|^\.|^\"|^\/|^\\|$\"','',string)
    new_string = string.split()
    string = ' '.join(new_string)
    string = re.sub('^:|^!|^\?|^\-|^\.|^\"|^\/|^\\|$\"','',string)
    new_string = string.split()
    string = ' '.join(new_string)
    
    if len(string) > 5:
        return string
    else:
        return 'remove'

def get_freq_label(list_labels):
    counting_labels = Counter(list_labels)
    most_frequent = counting_labels.most_common(1)[0][0]
    return most_frequent

def origin_id(aux):
    id_and_origin = aux.split('_')
    return id_and_origin[-1]

def get_id(aux):
    id_and_origin = aux.split('_')
    try:
        tweet_id = int(id_and_origin[0])
        return tweet_id
    except:
        return 'remove'

# %% loading dataset
dataset = load_dataset("hatexplain")

# %% creating series with labels, ids and splits

# labels
list_labels = []

for item in ['train','test','validation']:
    for i in range(len(dataset[item])):
        list_labels.append(dataset[item][i]['annotators']['label'])

series_labels = pd.Series(list_labels)

# ids
id_validation = pd.Series(dataset['validation']['id'])
id_test = pd.Series(dataset['test']['id'])
id_train = pd.Series(dataset['train']['id'])
series_id = pd.concat([id_train,id_test,id_validation],axis=0).reset_index(drop=True)
del id_validation
del id_test
del id_train
gc.collect()

# split
list_split_name = ['train'] * len(dataset['train']) + ['test'] * len(dataset['test']) + ['validation'] * len(dataset['validation'])
series_split_name = pd.Series(list_split_name)

# %% creating dataframe
df = pd.DataFrame([series_id,series_labels,series_split_name]).T
df.columns = ['id','list_label','split']
df['label'] = df['list_label'].apply(lambda x : get_freq_label(x)) # most frequent label is chosen
df['origin'] = df['id'].apply(lambda x : origin_id(x))
df = df[df['origin'] == 'twitter'] # only tweets will be retrieved
df['tweet_id'] = df['id'].apply(lambda x : get_id(x))
df = df[df['tweet_id'] != 'remove']
df = df[['tweet_id','label','split']]

# Retrieving tweets
# %%
import tweepy

# %% loading credentials
# TODO: Add twitter integration

# %% creating function to retrieve tweets based on their ids
def get_tweet_with_id(tweet_identifier):
    try:
        tweet = api.get_status(tweet_identifier)
        tweet_string_text = tweet.text
        return tweet_string_text
    except:
        return 'banned_user'

# %% changing file directory
os.chdir('/mnt/c/Users/Acer/Documents/Corpora/downloaded/HS14_Hatexplain')

# %% retrieving tweets based on their ids
df['text'] = df['tweet_id'].apply(lambda x: get_tweet_with_id(x))
df = df[['text','label','tweet_id','split']]

# %% saving file
df.to_csv('HS14_all_splits_id_with_text.csv',sep=';',index=False)