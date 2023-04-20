# -*- coding: utf-8 -*-
# %% bibtex

'''
@inproceedings{strapparava-mihalcea-2007-semeval_affective,
title = "{S}em{E}val-2007 Task 14: Affective Text",
author = "Strapparava, Carlo and Mihalcea, Rada",
booktitle = "Proceedings of the Fourth International Workshop on Semantic Evaluations ({S}em{E}val-2007)",
month = jun,
year = "2007",
address = "Prague, Czech Republic",
publisher = "Association for Computational Linguistics",
url = "https://www.aclweb.org/anthology/S07-1013",
pages = "70--74",
}
'''

# %% loading libraries
import pandas as pd
from tqdm import tqdm
import os
from bs4 import BeautifulSoup
import html
import tarfile

# %% loading file
os.chdir('/mnt/c/Users/Acer/Documents/Corpora/downloaded')
tar = tarfile.open("AffectiveText.Semeval.2007.tar.gz", "r:gz")

# %% function to create dataframes
def create_dataframes():
    
    file_names = ['AffectiveText.test/affectivetext_test.xml','AffectiveText.test/affectivetext_test.emotions.gold','AffectiveText.trial/affectivetext_trial.xml','AffectiveText.trial/affectivetext_trial.emotions.gold']
    list_sentiments = ['sentiment_00','sentiment_01','sentiment_02','sentiment_03','sentiment_04','sentiment_05']

    for i in range(len(file_names)):
#        print(i)

        if i == 0 or i == 2:
    #        print('if')
            list_files = []
            name = file_names[i]
            f = tar.extractfile(name)
            list_files.append(f.read())
            string_text = str(list_files[0])
            split_text = string_text.split('\\n')
            split_text = split_text[1:]

            for j in range(len(split_text)):   
                split_text[j] = str(split_text[j])
                split_text[j] = html.escape(split_text[j])
                split_text[j] = html.unescape(split_text[j])
                split_text[j] = BeautifulSoup(split_text[j], "lxml")
                split_text[j] = split_text[j].get_text()

            split_text = split_text[:-2]
            df_text = pd.DataFrame(split_text)


        elif i == 1 or i == 3:   
    #        print('elif label')
            list_files = []
            name = file_names[i]
            f = tar.extractfile(name)
            list_files.append(f.read())
            string_text = str(list_files[0])
            split_text = string_text.split('\\n')
            final_list = []

            for j in range(len(split_text)):   
                split_text[j] = str(split_text[j])
                split_text[j] = split_text[j].split()
                list_aux = []    
                for k in range(len(split_text[j])):
                    if len(split_text[k]) == 7:
                        if j == 0 and k != 0:
                            try:
                                list_aux.append(int(split_text[j][k]))
                            except:
                                list_aux.append(split_text[j][k])
                        elif j != 0:
                            try:
                                list_aux.append(int(split_text[j][k]))
                            except:
                                list_aux.append(split_text[j][k])
                    else:
                        list_aux.append(split_text[j][k])
                final_list.append(list_aux)
            final_list = final_list[:-1]
            df_label = pd.DataFrame(final_list)


            if i == 1:        
    #            print('elif test')
                df_test = pd.concat([df_text,df_label],axis=1)
                df_test.columns = ['text','index','sentiment_00','sentiment_01','sentiment_02','sentiment_03','sentiment_04','sentiment_05']
                df_test = df_test[['text','sentiment_00','sentiment_01','sentiment_02','sentiment_03','sentiment_04','sentiment_05']]
                # %% the most intense sentiment will be the label
    #            print(df_test)
                for sentiment in list_sentiments:
                    df_test[sentiment] = pd.to_numeric(df_test[sentiment])
    #            print('aqui')
                df_test['label'] = df_test[list_sentiments].idxmax(axis = 1)
                df_test = df_test[['text','label']]


            elif i == 3:
    #            print('elif trial')
                df_trial = pd.concat([df_text,df_label],axis=1)
                df_trial.columns = ['text','index','sentiment_00','sentiment_01','sentiment_02','sentiment_03','sentiment_04','sentiment_05']
                df_trial = df_trial[['text','sentiment_00','sentiment_01','sentiment_02','sentiment_03','sentiment_04','sentiment_05']]
                # %% the most intense sentiment will be the label
                for sentiment in list_sentiments:
                    df_trial[sentiment] = pd.to_numeric(df_trial[sentiment])
                df_trial['label'] = df_trial[list_sentiments].idxmax(axis = 1)
                df_trial = df_trial[['text','label']]
    return df_test, df_trial

# %% creating dataframes
df_train, df_test = create_dataframes()

# %% saving to csv multiclass dataframe
os.chdir('/mnt/c/Users/Acer/Documents/Corpora/Multiclass/Sentiment_Analysis')
df_train.to_csv('SA02_Affective_Text_train.csv',sep=';',index=False)
df_test.to_csv('SA02_Affective_Text_test.csv',sep=';',index=False)

# %% creating binary dataframes
os.chdir('/mnt/c/Users/Acer/Documents/Corpora/Binary/Sentiment_Analysis/Affective_Text')
unique_classes = sorted(df_train['label'].unique())

list_csv = []
for i in tqdm(range(len(unique_classes))):
    for j in range(i+1,len(unique_classes)):
        if i != j:
            # train
            df_aux = df_train.loc[(df_train['label'] == unique_classes[i]) | (df_train['label'] == unique_classes[j])]
            df_aux.to_csv(f'SA02_Affective_Text_Binary_{i}_{j}_train.csv',sep=';',index=False)
            list_csv.append([f'SA02_Affective_Text_Binary_{i}_{j}_train.csv',f'{unique_classes[i],unique_classes[j]}'])
            # test
            df_aux = df_test.loc[(df_test['label'] == unique_classes[i]) | (df_test['label'] == unique_classes[j])]
            df_aux.to_csv(f'SA02_Affective_Text_Binary_{i}_{j}_test.csv',sep=';',index=False)
            list_csv.append([f'SA02_Affective_Text_Binary_{i}_{j}_test.csv',f'{unique_classes[i],unique_classes[j]}'])

os.chdir('/mnt/c/Users/Acer/Documents/Corpora/Binary/Sentiment_Analysis/Explained')
df_list_csv = pd.DataFrame(list_csv,columns=['file_name','classes'])
df_list_csv.to_csv('SA02_Affective_Text_Binary_explained.csv',sep=';',index=False)