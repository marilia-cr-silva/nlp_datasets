"""
This .py file creates sentence embeddings for all the hate speech
datasets
"""
# %% loading libraries
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, TFAutoModel

# %%


def sent_transformer(name_lm: str, df_dataset: pd.DataFrame) -> pd.DataFrame:
    """
    This function embeds instances based on sentence transformers.
    input - param:
    name_lm (string):
    language model name
    df_dataset (pandas dataframe):
    dataframe with two columns. The first is the text column
    and the second has the labels

    output:
    df_final_embedding (pandas dataframe):
    dataframe with text (embedded) in the first column and the target (label)
    in the second column
    """
    language_model = SentenceTransformer(name_lm)
    sentences = list(df_dataset[NAME_TEXT_COLUMN])
    embedding = language_model.encode(sentences)
    embed_list = embedding.tolist()

    df_embed = pd.DataFrame({NAME_TEXT_COLUMN: embed_list})
    targets = df_dataset[NAME_TARGET_COLUMN]
    df_final_embedding = pd.concat((df_embed, targets), axis=1)
    df_final_embedding.columns = ["text", "label"]

    return df_final_embedding


# %%
os.chdir("./Hate_Speech")
list_selected_files = os.listdir()
NAME_TEXT_COLUMN = "text"
NAME_TARGET_COLUMN = "label"

for selected_file in list_selected_files:
    df_content = pd.read_csv(selected_file, sep=";")
    embed_df = sent_transformer(
        name_lm="sentence-transformers/all-MiniLM-L6-v2",
        df_dataset=df_content)
    name_pickle_file = selected_file[:-4]
    embed_df.to_pickle(f"embedded_{name_pickle_file}.pkl")
