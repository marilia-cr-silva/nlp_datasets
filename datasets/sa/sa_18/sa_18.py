'''
@misc{kaggle_samsung_ssd,
  author={Kaggle},
  title = {Samsung Internal SSD Reviews},
  howpublished = {\url{https://www.kaggle.com/datasets/abdulrahmanalqannas/ssd-reviews}},
  note = {Accessed: 2022-04-06},
  year={2020}
}
'''

# %% loading libraries
import pandas as pd
from tqdm import tqdm
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

# %% creating dataframe
os.chdir('/mnt/c/Users/Acer/Documents/Corpora/downloaded/SA18_SSD')

# %% train
df_train = pd.read_csv('ssd_reviews.csv')
df_train = df_train[['review_title','overall_review','rating_stars']]
df_train['text'] = df_train['review_title'] + ' ' + df_train['overall_review'] # concat the title and review
df_train = df_train[['text','rating_stars']]
df_train.columns = ['text','label']
df_train['text'] = df_train['text'].apply(lambda x: noise_mitigation(x))
df_train = df_train[df_train['text'] != 'remove']
df_train = df_train.drop_duplicates(subset=['text'],keep='first')
df_train = df_train.dropna(subset='label')
df_train = df_train.sample(frac=1,random_state=42).reset_index(drop=True)

# %% saving to csv multiclass dataframe
os.chdir('/mnt/c/Users/Acer/Documents/Corpora/Multiclass/Sentiment_Analysis')
df_train.to_csv('SA18_Multi_train.csv',sep=';',index=False)

# %% creating binary dataframes
os.chdir('/mnt/c/Users/Acer/Documents/Corpora/Binary/Sentiment_Analysis')
unique_classes = sorted(df_train['label'].unique())

list_file_name = []
list_label_01 = []
list_label_02 = []

for i in tqdm(range(len(unique_classes))):
    for j in range(i+1,len(unique_classes)):
        # train
        csv_file_name_train = f'SA18_Binary_{i}_{j}_train.csv'
        df_aux = df_train.loc[(df_train['label'] == unique_classes[i]) | (df_train['label'] == unique_classes[j])]
        df_aux.to_csv(csv_file_name_train,sep=';',index=False)
        list_file_name.append(csv_file_name_train)
        list_label_01.append(unique_classes[i])
        list_label_02.append(unique_classes[j])
        
# %% creating explanation csv file
os.chdir('/mnt/c/Users/Acer/Documents/Corpora/Binary/Sentiment_Analysis/Explained')
df_list_csv = pd.DataFrame([list_file_name,list_label_01,list_label_02]).T
df_list_csv.columns = ['file_name','label_01','label_02']
aux_csv = pd.read_csv('SA_Binary_explained.csv',sep=';')
df_final_csv = pd.concat([aux_csv,df_list_csv])
df_final_csv.to_csv('SA_Binary_explained.csv',sep=';',index=False)