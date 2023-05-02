# %%
"""
This .py file runs the AutoGluon AutoML System

@article{agtabular,
  title={AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data},
  author={Erickson, Nick and Mueller, Jonas and Shirkov, Alexander and Zhang, Hang and Larroy, Pedro and Li, Mu and Smola, Alexander},
  journal={arXiv preprint arXiv:2003.06505},
  year={2020}
}
"""
# %%
import os
import pandas as pd
import numpy as np
import sklearn.metrics
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    TFAutoModel,
)
from autogluon.tabular import TabularDataset, TabularPredictor

from sklearn.metrics import balanced_accuracy_score

# %%

# %%
NAME_TEXT_COLUMN = "text"
NAME_TARGET_COLUMN = "label"

def sent_transformer(name_lm: str, df_dataset: pd.DataFrame) -> list:
    """
    This function embeds instances based on sentence transformers.
    outputs:
    df_final_text (pandas dataframe):
    dataframe with text (natural language) in the first column and the target 2nd
    df_final_embedding (pandas dataframe):
    dataframe with text (embedded) in the first column and the target 2nd
    embed_list (list):
    a list with all the embedded instances
    # Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
    # https://arxiv.org/abs/1908.10084
    # it states that sentence BERT is more appropriate to clustering tasks
    # sentence-transformers/all-MiniLM-L6-v2 the most popular at Hugging Face website,
    # model in English without specific tasks
    # such as paraphrasing
    # the model is more compact, leading to a 384-dimensions embedding
    """
    language_model = SentenceTransformer(name_lm)
    sentences = list(df_dataset[NAME_TEXT_COLUMN])
    embedding = language_model.encode(sentences)
    embed_list = embedding.tolist()
    return embed_list


df_train = pd.read_csv("sa_01_bin_train_0_1.csv",sep=";")
df_test = pd.read_csv("sa_01_bin_test_0_1.csv",sep=";")

# %%
X_train = np.array(sent_transformer(
                    name_lm="sentence-transformers/all-MiniLM-L6-v2",
                    df_dataset=df_train))
y_train = np.array(df_train["label"])
# %%
X_test = np.array(df_test["text"])
X_test = np.array(sent_transformer(
                    name_lm="sentence-transformers/all-MiniLM-L6-v2",
                    df_dataset=df_test))
y_test = np.array(df_test["label"])

save_path = 'agModels-predictClass'  # specifies folder to store trained models
predictor = TabularPredictor(label="label").fit(df_train, time_limit=60, presets='high_quality')
y_pred = predictor.predict(df_test)

