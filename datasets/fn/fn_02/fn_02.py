"""
@inproceedings{Perez_Rosas18Automatic_fakenewsamt,
    author = {Verónica Pérez-Rosas and Bennett Kleinberg and Alexandra Lefevre and Rada Mihalcea},
    title = {\href{https://aclanthology.org/C18-1287/}{Automatic Detection of Fake News}},
    publisher = "Association for Computational Linguistics",
    journal = {COLING},
    address = "Santa Fe, New Mexico, USA",
    year = {2018}
}
https://lit.eecs.umich.edu/downloads.html#Fake%20News
http://web.eecs.umich.edu/~mihalcea/downloads/fakeNewsDatasets.zip
"""

# %%
import os
import pandas as pd

dataset_name = 'fnd02'

def goto_root():
    while os.getcwd().split('\\')[-1] != dataset_name.upper():
        os.chdir('../')

# %% processing fake dataset
goto_root()
os.chdir('./fakeNewsDataset')

# %% reading files
os.chdir('./fake')
fake_files = os.listdir()

# %% reading legit files
os.chdir('./legit')
legit_files = os.listdir()

# %% reading fake files content

goto_root()
os.chdir('./fakeNewsDataset/fake/')
fake_content = []
for f in fake_files:
    with open(f, 'r') as f:
        fake_content.append(f.read())


# %% reading legit files

goto_root()
os.chdir('./fakeNewsDataset/legit/')
legit_content = []
for f in legit_files:
    with open(f, 'r') as f:
        legit_content.append(f.read())


# %% creating dataframes

df_fake = pd.DataFrame(
    data={'text': fake_content, 'label': ['fake'] * len(fake_content)})
df_legit = pd.DataFrame(
    data={'text': legit_content, 'label': ['legit'] * len(legit_content)})

df_fake_news = df_fake.append(df_legit, ignore_index=True)

# %% saving csv
goto_root()
os.chdir('./fakeNewsDataset/')

df_train, df_test = train_test_split(
    df_fake_news, test_size=0.3, random_state=42,shuffle = True)

df_train.to_csv('fn_02_bin_train_0_1.csv', sep=';', index=False)
df_test.to_csv('fn_02_bin_test_0_1.csv', sep=';', index=False)

unique_classes = sorted(df_fake_news["label"].unique())
number_classes = len(unique_classes)

explained = {i: unique_classes[i] for i in range(number_classes)}
