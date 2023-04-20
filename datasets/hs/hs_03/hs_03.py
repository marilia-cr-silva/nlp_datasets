'''
@inproceedings{gibert2018hate,
    title = "{Hate Speech Dataset from a White Supremacy Forum}",
    author = "de Gibert, Ona  and
      Perez, Naiara  and
      Garc{'\i}a-Pablos, Aitor  and
      Cuadros, Montse",
    booktitle = "Proceedings of the 2nd Workshop on Abusive Language Online ({ALW}2)",
    month = oct,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W18-5102",
    doi = "10.18653/v1/W18-5102",
    pages = "11--20",
}
'''
# %%
import os
import pandas as pd

# %%
dataset_name = 'hs03'

# %%
# Utility function to nagivate back to dataset root
def goto_root():
    while os.getcwd().split('\\')[-1] != dataset_name.upper():
        os.chdir('../')

# %%
annotations_df = pd.read_csv('annotations_metadata.csv')

# %%
# Creating text column with empty string
annotations_df['text'] = [''] * annotations_df.shape[0]
# Creating partition column which informs whether a text is in the test set or not 
annotations_df['partition'] = [0] * annotations_df.shape[0]

# %% Get test files
goto_root()
os.chdir('./sampled_test/')
test_files = os.listdir()
goto_root()

# %% Get train files
goto_root()
os.chdir('./sampled_train/')
train_files = os.listdir()
goto_root()

# %%
for test_file in test_files:
    with open(f'sampled_test/{test_file}') as f:
        content = f.read()
        
    # Obtém o id do arquivo
    file_id = test_file.split('.txt')[0]
    
    # Obtém o índice correspondente no annotations_df
    idx = annotations_df.index[annotations_df['file_id'] == file_id]
    
    # Substituindo as celulas de text e partition
    annotations_df.loc[idx, 'text'] = content
    annotations_df.loc[idx, 'partition'] = 1

# %%
for train_file in train_files:
    with open(f'sampled_train/{train_file}', encoding='utf-8') as f:
        content = f.read()
        
    # Obtém o id do arquivo
    file_id = train_file.split('.txt')[0]
    
    # Obtém o índice correspondente no annotations_df
    idx = annotations_df.index[annotations_df['file_id'] == file_id]
    
    # Substituindo as celulas de text e partition
    annotations_df.loc[idx, 'text'] = content
    annotations_df.loc[idx, 'partition'] = 2 

# %%
print(annotations_df[annotations_df['partition'] == 1].shape[0], len(test_files))
print(annotations_df[annotations_df['partition'] == 2].shape[0], len(train_files))

# %%
df_test = annotations_df[annotations_df['partition'] == 1]
df_test = df_test[["text", "label"]]
df_test.to_csv('test.csv', sep=';', index=False)

# %%
df_train = annotations_df[annotations_df['partition'] == 2]
df_train = df_train[["text", "label"]]
df_train.to_csv('train.csv', sep=';', index=False)