'''
@inproceedings{ShahiNandini2020_FakeCovid,
  place = {US},
  title = {FakeCovid- A Multilingual Cross-domain Fact Check News Dataset for COVID-19},
  url = {https://doi.org/10.36190/2020.14},
  DOI = {10.36190/2020.14},
  publisher = {ICWSM},
  author = {Shahi,  Gautam Kishore and Nandini,  Durgesh},
  year = {2020},
  month = Jun,
}
'''

# %% loading libraries

import pandas as pd
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings("ignore")

# %% loading file and creating dataframe
os.chdir('/mnt/c/Users/Acer/Documents/Corpora/downloaded/FND06')
df = pd.read_csv('FND06_FakeCovid.csv')
df = df[df['lang'] == 'en']
df.dropna(subset=['ref_category_title'],inplace=True)
df = df[['title','class']]
df.columns = ['text','label']

# %% replacing labels
df['label'] = df['label'].replace('(Org. doesn\'t apply rating)','')
df['label'] = df['label'].replace('FALSE','False')
df['label'] = df['label'].replace('Fake','False')
df['label'] = df['label'].replace('HALF TRUE','Partly True')
df['label'] = df['label'].replace('IN DISPUTE','')
df['label'] = df['label'].replace('MISLEADING','Misleading')
df['label'] = df['label'].replace('MOSTLY FALSE','Mostly False')
df['label'] = df['label'].replace('MOSTLY TRUE','Mostly True')
df['label'] = df['label'].replace('Mixed','')
df['label'] = df['label'].replace('Mostly false','Mostly False')
df['label'] = df['label'].replace('No evidence','')
df['label'] = df['label'].replace('No Evidence','')
df['label'] = df['label'].replace('Not true','False')
df['label'] = df['label'].replace('PANTS ON FIRE','False')
df['label'] = df['label'].replace('PARTLY FALSE','Partly False')
df['label'] = df['label'].replace('PARTLY TRUE','Partly True')
df['label'] = df['label'].replace('Partially correct','Partly True')
df['label'] = df['label'].replace('Partially false','Partly False')
df['label'] = df['label'].replace('Partially true','Partly True')
df['label'] = df['label'].replace('Partly true','Partly True')
df['label'] = df['label'].replace('Partly false','Partly False')
df['label'] = df['label'].replace('Two Pinocchios','')
df['label'] = df['label'].replace('Unlikely','')
df['label'] = df['label'].replace('half true','Partly True')
df['label'] = df['label'].replace('Half True','Partly True')
df['label'] = df['label'].replace('mislEADING','Misleading')
df['label'] = df['label'].replace('misleading','Misleading')
# due to the smaller number of true labels ('Partly True' (15) and 'Mostly True' (11)), they'll be merged
df['label'] = df['label'].replace('Partly True','True')
df['label'] = df['label'].replace('Mostly True','True')

# removing empty labels
df = df[df['label'] != '']
# removing duplicate instances
df = df.drop_duplicates(subset=['text'],keep='first')
df.reset_index(inplace=True,drop=True)

# %%
df_train, df_test = train_test_split(
    df, test_size=0.3, random_state=42,shuffle = True)

df_train.to_csv(f"fn_08_multi_train.csv",sep=";",index=False)
df_test.to_csv(f"fn_08_multi_test.csv",sep=";",index=False)

# %%
unique_classes = sorted(df_train["label"].unique())

list_csv = []
for i in tqdm(range(len(unique_classes))):
    for j in range(i+1,len(unique_classes)):
        # train
        df_aux = df_train.loc[(df_train["label"] == unique_classes[i]) | (df_train["label"] == unique_classes[j])]
        df_aux.to_csv(f"fn_08_bin_train_{i}_{j}.csv",sep=";",index=False)
        list_csv.append([f"fn_08_bin_train_{i}_{j}.csv",f"{unique_classes[i],unique_classes[j]}"])
        # test
        df_aux = df_test.loc[(df_test["label"] == unique_classes[i]) | (df_test["label"] == unique_classes[j])]
        df_aux.to_csv(f"fn_08_bin_test_{i}_{j}.csv",sep=";",index=False)
        list_csv.append([f"fn_08_bin_test_{i}_{j}.csv",f"{unique_classes[i],unique_classes[j]}"])

df_list_csv = pd.DataFrame(list_csv,columns=["file_name","classes"])
df_list_csv.to_csv("fn_08_bin_explained.csv",sep=";",index=False)

number_classes = unique_classes
explained = {i: unique_classes[i] for i in range(number_classes)}