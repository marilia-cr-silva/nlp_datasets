'''
@inproceedings{marc_reviews_amazon_multi,
    title={The Multilingual Amazon Reviews Corpus},
    author={Keung, Phillip and Lu, Yichao and Szarvas, György and Smith, Noah A.},
    booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing},
    year={2020}
}
'''

# %% loading libraries
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import os
import re
import html
import fasttext
from bs4 import BeautifulSoup
#from zipfile import ZipFile
#import tarfile
import csv
import io
import warnings
warnings.filterwarnings("ignore")

# %% loading language detection model
language_identification_model = fasttext.load_model('lid.176.bin') # or lid.176.ftz lid.176.bin

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
string = re.sub('\n|\t','',string)
    
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
string = re.sub('\n|\t','',string)
    
    return string


# %% function to identify language
def detect_language(instance):
    
    aux = str(language_identification_model.predict(instance, k=1)[0][0][-2:])
    
    return aux

# %% loading dataset
dataset = load_dataset("amazon_reviews_multi",'en')

# %% creating dataframes
df_test_aux = pd.DataFrame([dataset['test']['review_body'],dataset['test']['stars'],dataset['test']['language']]).T
df_validation_aux = pd.DataFrame([dataset['validation']['review_body'],dataset['validation']['stars'],dataset['validation']['language']]).T
df_test = pd.concat([df_validation_aux,df_test_aux])
df_test.columns = ['text','label','language']
df_test = df_test[df_test['language'] == 'en']
df_test = df_test[['text','label']]
df_train = pd.DataFrame([dataset['train']['review_body'],dataset['train']['stars'],dataset['train']['language']]).T
df_train.columns = ['text','label','language']
df_train = df_train[df_train['language'] == 'en']
df_train = df_train[['text','label']]


# %% train set
df_train['text'] = df_train['text'].apply(lambda x: noise_mitigation(x))
#df_train['language'] = df_train['text'].apply(lambda x: detect_language(x))
#df_train = df_train[df_train['language'] == 'en']
df_train['text'] = df_train['text'].apply(lambda x: noise_mitigation_lang_id(x))
df_train = df_train.drop_duplicates(subset=['text'],keep='first')
df_train = df_train.sample(frac=1,random_state=42).reset_index(drop=True)[['text','label']]


# %% test set
df_test['text'] = df_test['text'].apply(lambda x: noise_mitigation(x))
#df_test['language'] = df_test['text'].apply(lambda x: detect_language(x))
#df_test = df_test[df_test['language'] == 'en']
df_test['text'] = df_test['text'].apply(lambda x: noise_mitigation_lang_id(x))
df_test = df_test.drop_duplicates(subset=['text'],keep='first')
df_test = df_test.sample(frac=1,random_state=42).reset_index(drop=True)[['text','label']]

# %% saving to csv multiclass dataframe
os.chdir('/mnt/c/Users/Acer/Documents/Corpora/Multiclass/Sentiment_Analysis')
df_train.to_csv('SA13_Amazon_Reviews_Multi_train.csv',sep=';',index=False)
df_test.to_csv('SA13_Amazon_Reviews_Multi_test.csv',sep=';',index=False)

# %% creating binary dataframes
os.chdir('/mnt/c/Users/Acer/Documents/Corpora/Binary/Sentiment_Analysis/Amazon_Reviews_Multi')
unique_classes = sorted(df_train['label'].unique())

list_csv = []
for i in tqdm(range(len(unique_classes))):
    for j in range(i+1,len(unique_classes)):
        if i != j:
            # train
            df_aux = df_train.loc[(df_train['label'] == unique_classes[i]) | (df_train['label'] == unique_classes[j])]
            df_aux.to_csv(f'SA13_Amazon_Reviews_Multi_Binary_train_{i+1}_{j+1}.csv',sep=';',index=False)
            list_csv.append([f'SA13_Amazon_Reviews_Multi_Binary_train_{i+1}_{j+1}.csv',f'{unique_classes[i],unique_classes[j]}'])
            # test
            df_aux = df_test.loc[(df_test['label'] == unique_classes[i]) | (df_test['label'] == unique_classes[j])]
            df_aux.to_csv(f'SA13_Amazon_Reviews_Multi_Binary_test_{i+1}_{j+1}.csv',sep=';',index=False)
            list_csv.append([f'SA13_Amazon_Reviews_Multi_Binary_test_{i+1}_{j+1}.csv',f'{unique_classes[i],unique_classes[j]}'])

os.chdir('/mnt/c/Users/Acer/Documents/Corpora/Binary/Sentiment_Analysis/Explained')
df_list_csv = pd.DataFrame(list_csv,columns=['file_name','classes'])
df_list_csv.to_csv('SA13_Amazon_Reviews_Multi_Binary_explained.csv',sep=';',index=False)