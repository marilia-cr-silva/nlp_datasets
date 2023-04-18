"""
@inproceedings{torabi-asr-taboada-2018-data-misinfotext,
    title = "The Data Challenge in Misinformation Detection: Source Reputation vs. Content Veracity",
    author = "Torabi Asr, Fatemeh  and
      Taboada, Maite",
    booktitle = "Proceedings of the First Workshop on Fact Extraction and {VER}ification ({FEVER})",
    month = nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W18-5502",
    doi = "10.18653/v1/W18-5502",
    pages = "10--15",
    abstract = "Misinformation detection at the level of full news articles is a text classification problem. Reliably labeled data in this domain is rare. Previous work relied on news articles collected from so-called {``}reputable{''} and {``}suspicious{''} websites and labeled accordingly. We leverage fact-checking websites to collect individually-labeled news articles with regard to the veracity of their content and use this data to test the cross-domain generalization of a classifier trained on bigger text collections but labeled according to source reputation. Our results suggest that reputation-based classification is not sufficient for predicting the veracity level of the majority of news articles, and that the system performance on different test datasets depends on topic distribution. Therefore collecting well-balanced and carefully-assessed training data is a priority for developing robust misinformation detection systems.",
}
"""

# %% loading libraries

import os
import pandas as pd
from tqdm import tqdm

dataset_name = 'fn_04'

df_content = pd.read_csv(
    'snopes_phase2_raw_2018_7_3.csv',
    sep=',',
    lineterminator='\n',
    usecols=['fact_rating_phase1', 'snopes_url_phase1', 'article_title_phase1', 'article_category_phase1', 'article_date_phase1', 'article_claim_phase1', 'article_origin_url_phase1', 'index_paragraph_phase1', 'page_is_first_citation_phase1', 'error_phase2', 'original_article_text_phase2', 'article_title_phase2', 'publish_date_phase2', 'author_phase2'])

df_content = df_content[['article_claim_phase1', 'fact_rating_phase1']]
df_content.rename(
    {
        'article_claim_phase1': 'text',
        'fact_rating_phase1': 'label'
    },
    inplace=True,
    axis=1)

# %%

df_content.to_csv('train.csv', sep=';', index=False)

# Garantindo que os diretorios de split para treino existe
folders = ['train_split']
files_in_curr_dir = os.listdir()
for folder in folders:
    if folder not in files_in_curr_dir:
        os.mkdir(folder)

# %%
df_train, df_test = train_test_split(
    df_content, test_size=0.3, random_state=42,shuffle = True)

df_train.to_csv(f"fn_04_multi_train.csv",sep=";",index=False)
df_test.to_csv(f"fn_04_multi_test.csv",sep=";",index=False)

# %%
unique_classes = sorted(df_train["label"].unique())

list_csv = []
for i in tqdm(range(len(unique_classes))):
    for j in range(i+1,len(unique_classes)):
        # train
        df_aux = df_train.loc[(df_train["label"] == unique_classes[i]) | (df_train["label"] == unique_classes[j])]
        df_aux.to_csv(f"fn_04_bin_train_{i}_{j}.csv",sep=";",index=False)
        list_csv.append([f"fn_04_bin_train_{i}_{j}.csv",f"{unique_classes[i],unique_classes[j]}"])
        # test
        df_aux = df_test.loc[(df_test["label"] == unique_classes[i]) | (df_test["label"] == unique_classes[j])]
        df_aux.to_csv(f"fn_04_bin_test_{i}_{j}.csv",sep=";",index=False)
        list_csv.append([f"fn_04_bin_test_{i}_{j}.csv",f"{unique_classes[i],unique_classes[j]}"])

df_list_csv = pd.DataFrame(list_csv,columns=["file_name","classes"])
df_list_csv.to_csv("fn_04_bin_explained.csv",sep=";",index=False)

number_classes = unique_classes
explained = {i: unique_classes[i] for i in range(number_classes)}