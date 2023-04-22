'''
"@inproceedings{sheng-uthus-2020-investigating-poem,
    title = ""Investigating Societal Biases in a Poetry Composition System"",
    author = ""Sheng, Emily  and
      Uthus, David"",
    booktitle = ""Proceedings of the Second Workshop on Gender Bias in Natural Language Processing"",
    month = dec,
    year = ""2020"",
    address = ""Barcelona, Spain (Online)"",
    publisher = ""Association for Computational Linguistics"",
    url = ""https://aclanthology.org/2020.gebnlp-1.9"",
    pages = ""93--106"",
}"
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
import warnings
warnings.filterwarnings("ignore")

# %% loading language detection model
language_identification_model = fasttext.load_model('/mnt/c/Users/Acer/Documents/Corpora/fasttext/lid.176.bin') # or lid.176.ftz lid.176.bin

# %% function to reduce the noise before language identification
def noise_mitigation(aux):
    
    #string = aux.decode('utf-8')
    #string = str(string)
    string = str(aux)
    new_string = string.split('\n')
    string = ' '.join(new_string)
    string = re.sub('\s\#\s|\@user\s?','',string)
    string = re.sub('\-\-+|\s\-\s',' ',string)
    string = re.sub('\s?\@\s',' at ',string)
    string = re.sub(r'(https?:(\/\/))?(www\.)?(\w+)\.(\w+)(\.?\/?)+([a-zA-Z0–9@:%&._\+-?!~#=]*)?(\.?\/)?([a-zA-Z0–9@:%&._\/+-~?!#=]*)?(\.?\/)?',' ',string) # websites
    string = re.sub('w\/|W\/','with',string)
    
#    string = html.escape(string)
#    string = html.unescape(string)
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

# %% loading dataset
dataset = load_dataset('poem_sentiment')

# %% creating dataframes
df_test_aux = pd.DataFrame([dataset['test']['verse_text'],dataset['test']['label']]).T
df_validation_aux = pd.DataFrame([dataset['validation']['verse_text'],dataset['validation']['label']]).T
df_test = pd.concat([df_validation_aux,df_test_aux])
df_test.columns = ['text','label']
df_train = pd.DataFrame([dataset['train']['verse_text'],dataset['train']['label']]).T
df_train.columns = ['text','label']

# %% train set
df_train['text'] = df_train['text'].apply(lambda x: noise_mitigation(x))
#df_train['language'] = df_train['text'].apply(lambda x: detect_language(x))
#df_train = df_train[df_train['language'] == 'en']
df_train['text'] = df_train['text'].apply(lambda x: noise_mitigation_lang_id(x))
df_train = df_train.drop_duplicates(subset=['text'],keep='first')
#df_train = df_train.sample(frac=1,random_state=42).reset_index(drop=True)[['text','label']]
df_train = df_train.reset_index(drop=True)[['text','label']]

# %% test set
df_test['text'] = df_test['text'].apply(lambda x: noise_mitigation(x))
#df_test['language'] = df_test['text'].apply(lambda x: detect_language(x))
#df_test = df_test[df_test['language'] == 'en']
df_test['text'] = df_test['text'].apply(lambda x: noise_mitigation_lang_id(x))
df_test = df_test.drop_duplicates(subset=['text'],keep='first')
#df_test = df_test.sample(frac=1,random_state=42).reset_index(drop=True)[['text','label']]
df_test = df_test.reset_index(drop=True)[['text','label']]

# %% saving to csv multiclass dataframe
os.chdir('/mnt/c/Users/Acer/Documents/Corpora/Multiclass/Sentiment_Analysis')
df_train.to_csv(f'SA17_Poem_Sentiment_Multi_train.csv',sep=';',index=False)
df_test.to_csv(f'SA17_Poem_Sentiment_Multi_test.csv',sep=';',index=False)

# %% creating binary dataframes
os.chdir('/mnt/c/Users/Acer/Documents/Corpora/Binary/Sentiment_Analysis/Poem_Sentiment')
unique_classes = sorted(df_train['label'].unique())

list_csv = []
for i in tqdm(range(len(unique_classes))):
    for j in range(i+1,len(unique_classes)):
        if i != j:
            # train
            df_aux = df_train.loc[(df_train['label'] == unique_classes[i]) | (df_train['label'] == unique_classes[j])]
            df_aux.to_csv(f'SA17_Poem_Sentiment_Binary_train_{i}_{j}.csv',sep=';',index=False)
            list_csv.append([f'SA17_Poem_Sentiment_Binary_train_{i}_{j}.csv',f'{unique_classes[i],unique_classes[j]}'])
            # test
            df_aux = df_test.loc[(df_test['label'] == unique_classes[i]) | (df_test['label'] == unique_classes[j])]
            df_aux.to_csv(f'SA17_Poem_Sentiment_Binary_test_{i}_{j}.csv',sep=';',index=False)
            list_csv.append([f'SA17_Poem_Sentiment_Binary_test_{i}_{j}.csv',f'{unique_classes[i],unique_classes[j]}'])

os.chdir('/mnt/c/Users/Acer/Documents/Corpora/Binary/Sentiment_Analysis/Explained')
df_list_csv = pd.DataFrame(list_csv,columns=['file_name','classes'])
df_list_csv.to_csv('SA17_Poem_Sentiment_Binary_explained.csv',sep=';',index=False)