"""
@misc{abu_salem_fatima_k_2019_fake_news_syrian,
  author       = {Abu Salem, Fatima K. and
                  Al Feel, Roaa and
                  Elbassuoni, Shady and
                  Jaber, Mohamad and
                  Farah, May},
  title        = {\href{https://doi.org/10.5281/zenodo.2607278}{FA-KES: A Fake News Dataset around the Syrian War}},
  month        = jan,
  year         = 2019,
  doi          = {10.5281/zenodo.2607278},
}
"""

import pandas as pd
import json
import os
from tqdm import tqdm

dataset_name = 'fn_06'
train_filename = 'FA-KES-Dataset.csv'

df_content = pd.read_csv(train_filename, encoding='latin-1')
df_content = df_content[["article_content", "labels"]]
df_content.rename(
    {
        'article_content': 'text',
        'labels': 'label'
    }, 
    axis=1, 
    inplace=True)

df_train, df_test = train_test_split(
    df_content, test_size=0.3, random_state=42,shuffle = True)

df_train.to_csv('fn_06_bin_train_0_1.csv', sep=';', index=False)
df_test.to_csv('fn_06_bin_test_0_1.csv', sep=';', index=False)

unique_classes = sorted(df_content["label"].unique())
number_classes = len(unique_classes)

explained = {i: unique_classes[i] for i in range(number_classes)}