'''
@inproceedings{10.1145/3194658.3194677_druglib_data,
author = {Gr\"{a}\ss{}er, Felix and Kallumadi, Surya and Malberg, Hagen and Zaunseder, Sebastian},
title = {Aspect-Based Sentiment Analysis of Drug Reviews Applying Cross-Domain and Cross-Data Learning},
year = {2018},
isbn = {9781450364935},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3194658.3194677},
doi = {10.1145/3194658.3194677},
abstract = {Online review sites and opinion forums contain a wealth of information regarding user preferences and experiences over multiple product domains. This information can be leveraged to obtain valuable insights using data mining approaches such as sentiment analysis. In this work we examine online user reviews within the pharmaceutical field. Online user reviews in this domain contain information related to multiple aspects such as effectiveness of drugs and side effects, which make automatic analysis very interesting but also challenging. However, analyzing sentiments concerning the various aspects of drug reviews can provide valuable insights, help with decision making and improve monitoring public health by revealing collective experience. In this preliminary work we perform multiple tasks over drug reviews with data obtained by crawling online pharmaceutical review sites. We first perform sentiment analysis to predict the sentiments concerning overall satisfaction, side effects and effectiveness of user reviews on specific drugs. To meet the challenge of lacking annotated data we further investigate the transferability of trained classification models among domains, i.e. conditions, and data sources. In this work we show that transfer learning approaches can be used to exploit similarities across domains and is a promising approach for cross-domain sentiment analysis.},
booktitle = {Proceedings of the 2018 International Conference on Digital Health},
pages = {121–125},
numpages = {5},
keywords = {clinical decision support system (cdss), sentiment analysis, health recommender system, text classification},
location = {Lyon, France},
series = {DH '18}

https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Druglib.com%29
https://archive.ics.uci.edu/ml/machine-learning-databases/00461/drugLib_raw.zip
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
from zipfile import ZipFile
#import tarfile
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
    string = re.sub('\&','and',string)

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

# %% loading files - hierarchy - part 1
file_name = "drugLib_raw.zip"
with ZipFile(file_name, 'r') as zip:
    zip.extractall()

# %% train set
df_train = pd.read_csv('drugLibTrain_raw.tsv',sep='\t')
df_train = df_train[['commentsReview','rating']]
df_train.columns = ['text','label']
df_train['text'] = df_train['text'].apply(lambda x: noise_mitigation(x))
df_train['language'] = df_train['text'].apply(lambda x: detect_language(x))
df_train = df_train[df_train['language'] == 'en']
df_train['text'] = df_train['text'].apply(lambda x: noise_mitigation_lang_id(x))
df_train = df_train.drop_duplicates(subset=['text'],keep='first')
df_train = df_train.sample(frac=1,random_state=42).reset_index(drop=True)[['text','label']]

# %% test set
df_test = pd.read_csv('drugLibTest_raw.tsv',sep='\t')
df_test = df_test[['commentsReview','rating']]
df_test.columns = ['text','label']
df_test['text'] = df_test['text'].apply(lambda x: noise_mitigation(x))
df_test['language'] = df_test['text'].apply(lambda x: detect_language(x))
df_test = df_test[df_test['language'] == 'en']
df_test['text'] = df_test['text'].apply(lambda x: noise_mitigation_lang_id(x))
df_test = df_test.drop_duplicates(subset=['text'],keep='first')
df_test = df_test.sample(frac=1,random_state=42).reset_index(drop=True)[['text','label']]

# %% saving to csv multiclass dataframe
os.chdir('/mnt/c/Users/Acer/Documents/Corpora/Multiclass/Sentiment_Analysis')
df_train.to_csv('SA09_Drug_Review_Dataset_train.csv',sep=';',index=False)
df_test.to_csv('SA09_Drug_Review_Dataset_test.csv',sep=';',index=False)

# %% creating binary dataframes
os.chdir('/mnt/c/Users/Acer/Documents/Corpora/Binary/Sentiment_Analysis/Drug_Review_Dataset')
unique_classes = sorted(df_train['label'].unique())

list_csv = []
for i in tqdm(range(len(unique_classes))):
    for j in range(i+1,len(unique_classes)):
        if i != j:
            # train
            df_aux = df_train.loc[(df_train['label'] == unique_classes[i]) | (df_train['label'] == unique_classes[j])]
            df_aux.to_csv(f'SA09_Drug_Review_Dataset_Binary_{i+1}_{j+1}_train.csv',sep=';',index=False)
            list_csv.append([f'SA09_Drug_Review_Dataset_Binary_{i+1}_{j+1}_train.csv',f'{unique_classes[i],unique_classes[j]}'])
            # test
            df_aux = df_test.loc[(df_test['label'] == unique_classes[i]) | (df_test['label'] == unique_classes[j])]
            df_aux.to_csv(f'SA09_Drug_Review_Dataset_Binary_{i+1}_{j+1}_test.csv',sep=';',index=False)
            list_csv.append([f'SA09_Drug_Review_Dataset_Binary_{i+1}_{j+1}_test.csv',f'{unique_classes[i],unique_classes[j]}'])

os.chdir('/mnt/c/Users/Acer/Documents/Corpora/Binary/Sentiment_Analysis/Explained')
df_list_csv = pd.DataFrame(list_csv,columns=['file_name','classes'])
df_list_csv.to_csv('SA09_Drug_Review_Dataset_Binary_explained.csv',sep=';',index=False)