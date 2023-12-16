"""
This .py file extracts the meta-features of each dataset.
"""

# %% loading libraries
import gc
import os
import re
#!pip3 install demoji
import demoji
import numpy as np
import pandas as pd
#!pip3 install spacy
import spacy
#!pip3 install textstat
import textstat
from scipy.stats import entropy
from sklearn.decomposition import PCA
from spacy.lang.en import English

# %%
demoji.download_codes()
English = spacy.load("en_core_web_sm")


# %% functions

def get_number_tokens(row: pd.Series) -> int:
    """
    It returns the number of tokens per instance.

    input - param:
    text_instance (str):
    instance that is going to be tokenized
    and whose number of tokens are going to be counted
    output:
    (int) number of tokens
    """
    text_instance = row["text"]
    list_tokens = English(text_instance)
    return len(list_tokens)


def get_number_unique_tokens(row: pd.Series) -> int:
    """
    It returns the number of unique tokens per instance.

    input - param:
    row (pd.Series):
    the row that is going to be processed
    and whose number of unique tokens are going to be counted
    output:
    (int) number of unique tokens
    """
    text_instance = row["text"]
    list_tokens = English(text_instance)
    list_unique = []

    for token in list_tokens:
        if token not in list_unique:
            list_unique.append(token)

    return len(list_unique)


def get_vocab_length(df_dataset: pd.DataFrame) -> int:
    """
    This function returns the total number of
    unique tokens in the entire dataset

    input:
    df_dataset (pandas dataframe):
    the dataframe that is going to be evaluated

    output:
    (int) number of unique words in the corpus
    """
    list_text = df_dataset["text"].tolist()
    list_vocabulary = []

    for text in list_text:
        list_tokens = English(text)

        for item in list_tokens:
            if item not in list_vocabulary:
                list_vocabulary.append(item)

    return len(list_vocabulary)


def get_average_word_length(row: pd.Series) -> float:
    """
    This function returns the average word length of the instance

    input - param:
    row (pd.Series):
    the row that is going to be processed
    and whose average word length is going to be calculated
    output:
    (float) the average word length
    """
    text_instance = row["text"]
    list_tokens = English(text_instance)
    avg_aux = 0
    for token in list_tokens:
        avg_aux += len(token)
    avg_length = avg_aux/len(list_tokens)
    return avg_length


def get_pos(row: pd.Series) -> list:
    """
    This function calculates the number of PoS tags
    per text instance.
    In the end it checks which PoS tags have
    representatives amongst all the possible tags

    input - param:
    row (pd.Series):
    the row that is going to be processed

    output:
    list_pos_count (list):
    list with the number of tokens per PoS
    """
    text_instance = row["text"]
    text_tokens = English(text_instance)
    words = []
    pos_tag = []
    for word in text_tokens:
        words.append(word.text)
        pos_tag.append(word.pos_)

    df_corpus_pos = pd.DataFrame([words, pos_tag]).T
    df_corpus_pos.columns = ["words", "pos_tag"]
    df_corpus_pos_count = df_corpus_pos["pos_tag"].value_counts()
    list_pos_tagging = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
                        "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]
    list_pos_count = [0]*len(list_pos_tagging)

    length_pos_tagging = len(df_corpus_pos_count)
    for i, tag_pos in enumerate(list_pos_tagging):
        for j in range(length_pos_tagging):
            if df_corpus_pos_count.index[j] == tag_pos:
                list_pos_count[i] = df_corpus_pos_count.iloc[j]
    return list_pos_count


def get_number_emojis(row: pd.Series) -> list:
    """
    This function returns a list with two items.
    The first is the text without emojis and the
    second item is the number of emojis.

    input - param:
    row (pd.Series):
    the row that is going to be processed

    output:
    list with the text without emoji and the number of emojis
    """
    string = row["text"]
    count_emojis = 0
    tokens_string = string.split()

    for i in range(len(tokens_string)):

        if tokens_string[i] != "":
            aux = demoji.findall(tokens_string[i])
            size = len(aux)

            if size > 0:  # it removes emojis
                count_emojis = count_emojis + size
                for key in aux.keys():
                    tokens_string[i] = re.sub(key, "", tokens_string[i])
    string = " ".join(tokens_string).split()
    string = " ".join(string)
    return [string, count_emojis]


def get_number_syllables(row: pd.Series) -> int:
    """
    This function returns the number of syllables of an excerpt.

    input - param:
    row (pd.series):
    a dataframe row
    output:
    summed_syllables (int):
    number of syllables of an instance
    """
    string = row["text"]
    list_tokens = English(string)
    summed_syllables = 0
    number_syllables = 0
    for item in list_tokens:
        number_syllables = textstat.syllable_count(str(item))
        summed_syllables = summed_syllables + number_syllables
    return summed_syllables


def get_count_special_characters(row: pd.Series) -> int:
    """
    This function returns the number the number of special characters
    per instance.

    input - param:
    row (pd.series):
    dataframe row that is going to be evaluated

    output:
    number_special_characters (int):
    number of special characters per instance
    """
    string = row["text"]
    pattern = re.compile(
        r"[\"\#\$\%\&\"\(\)\*\+\-\<\=\>\@\[\]\^\_\`\{\|\}\~\"]")
    list_found_special_characters = pattern.findall(string)
    number_special_characters = len(list_found_special_characters)
    return number_special_characters


def get_number_characters(row: pd.Series) -> int:
    """
    this function returns the number of characters per text instance
    input - param:
    row (pd.Series):
    the row that is going to be evaluated
    output:
    number_characters (int):
    the number of characters
    """
    number_characters = len(row["text"])
    return number_characters


def get_features_textstat(row: pd.Series) -> list:
    """
    this function returns text features extracted with textstat library.
    input - param:
    row (pd.Series):
    the row that is going to be evaluated
    output:
    list with all the collected metrics about readability, grade level,
    amongst others
    """
    instance_text = row["text"]
    value_flesch_reading_ease = textstat.flesch_reading_ease(instance_text)
    value_smog_index = textstat.smog_index(instance_text)
    value_flesch_kincaid_grade = textstat.flesch_kincaid_grade(instance_text)
    value_coleman_liau_index = textstat.coleman_liau_index(instance_text)
    value_dale_chall_readability_score = textstat.dale_chall_readability_score(
        instance_text)
    value_gunning_fog = textstat.gunning_fog(instance_text)
    value_automated_readability_index = textstat.automated_readability_index(
        instance_text)
    value_linsear_write_formula = textstat.linsear_write_formula(instance_text)

    return [value_flesch_reading_ease, value_smog_index, value_flesch_kincaid_grade,
            value_coleman_liau_index, value_dale_chall_readability_score, value_gunning_fog,
            value_automated_readability_index, value_linsear_write_formula]


# %% loading file
df_corpus = pd.read_csv("sa_01_bin_train_0_1.csv", sep=";")
df_test = pd.read_csv("sa_01_bin_test_0_1.csv", sep=";")

df_corpus["number_characters"] = df_corpus.apply(get_number_characters, axis=1)
number_instances_train = int(len(df_corpus))
number_instances_test = int(len(df_test))
df_test = ""

del df_test
gc.collect()

total_number_instances = int(number_instances_train + number_instances_test)
list_number_classes = list(df_corpus["label"].value_counts())
number_labels = int(len(list_number_classes))
class_imbalance = max(list_number_classes) / \
    min(list_number_classes)

df_corpus[["New_Text", "number_emojis"]] = df_corpus.apply(get_number_emojis, axis=1,
                                                           result_type="expand")
df_corpus.drop(columns=["text"], axis=1, inplace=True)
df_corpus.rename(columns={"New_Text": "text"}, inplace=True)

df_corpus["number_tokens_per_document"] = df_corpus.apply(
    get_number_tokens, axis=1)
df_corpus["count_syllables"] = df_corpus.apply(get_number_syllables, axis=1)


# %% Information Theoretic
df_corpus["number_unique_tokens"] = df_corpus.apply(
    get_number_unique_tokens, axis=1)
entropy_vocabulary_distribution = entropy(
    df_corpus["number_unique_tokens"])
entropy_words_per_document = entropy(
    df_corpus["number_tokens_per_document"])


# %% Vocabulary
vocabulary = get_vocab_length(df_corpus)
vocabulary_document_ratio = (df_corpus["number_unique_tokens"].max(
))/(df_corpus["number_unique_tokens"].min())
df_corpus["average_word_length"] = df_corpus.apply(
    get_average_word_length, axis=1)

# %% Readability
df_corpus[["flesch_reading_ease", "smog_index", "flesch_kincaid_grade", "coleman_liau_index",
           "dale_chall_readability_score", "gunning_fog", "automated_readability_index",
           "linsear_write_formula"]] = df_corpus.apply(
    get_features_textstat, axis=1, result_type="expand")

# %% Lexical Features
df_corpus[["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
           "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]] = df_corpus.apply(
    get_pos, axis=1, result_type="expand")

# %% PCA
df_corpus.drop(["text", "label"], axis=1, inplace=True)
auxiliary_array = np.array(df_corpus)
pca = PCA(random_state=0)
pca.fit(auxiliary_array)

del auxiliary_array
gc.collect()

pca_explained_variance = pca.explained_variance_
pca_explained_variance_ratio = pca.explained_variance_ratio_
df_corpus_pca = pd.DataFrame(
    [pca_explained_variance, pca_explained_variance_ratio]).T
df_corpus_pca.columns = ["pca_explained_variance",
                         "pca_explained_variance_ratio"]
pca_noise_variance = pca.noise_variance_
df_corpus = pd.concat([df_corpus, df_corpus_pca], axis=1)


# %% definition of the Minimum, Maximum, Average, Standard Deviation, Skewness, and Kurtosis
list_metrics = ["number_tokens_per_document", "number_unique_tokens",
                "average_word_length", "count_syllables", "flesch_reading_ease",
                "smog_index", "flesch_kincaid_grade", "coleman_liau_index",
                "dale_chall_readability_score", "gunning_fog",
                "automated_readability_index", "linsear_write_formula", "ADJ",
                "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
                "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM",
                "VERB", "X", "number_emojis", "pca_explained_variance",
                "pca_explained_variance_ratio"]

name_metric = []
score_metric = []

for meta_feat_metric in list_metrics:
    name_metric.append(f"minimum_{meta_feat_metric}")
    name_metric.append(f"maximum_{meta_feat_metric}")
    name_metric.append(f"average_{meta_feat_metric}")
    name_metric.append(f"standard_deviation_{meta_feat_metric}")
    name_metric.append(f"skewness_{meta_feat_metric}")
    name_metric.append(f"kurtosis_{meta_feat_metric}")

    score_metric.append(df_corpus[meta_feat_metric].min())
    score_metric.append(df_corpus[meta_feat_metric].max())
    score_metric.append(df_corpus[meta_feat_metric].mean())
    score_metric.append(df_corpus[meta_feat_metric].std())
    score_metric.append(df_corpus[meta_feat_metric].skew())
    score_metric.append(df_corpus[meta_feat_metric].kurt())

ratio_mean_std_dev_tokens = score_metric[2]/score_metric[3]
ratio_mean_std_dev_unique_tokens = score_metric[8]/score_metric[9]


# %% creating final df_corpus - part 01
name_columns = ["total_number_instances", "number_instances_train",
                "number_instances_test", "number_labels",
                "class_imbalance",
                "entropy_vocabulary_distribution", "entropy_words_per_document",
                "vocabulary", "vocabulary_document_ratio", "ratio_mean_std_dev_tokens",
                "ratio_mean_std_dev_unique_tokens", "pca_noise_variance"]

value_variables = [total_number_instances, number_instances_train,
                   number_instances_test, number_labels,
                   class_imbalance,
                   entropy_vocabulary_distribution, entropy_words_per_document,
                   vocabulary, vocabulary_document_ratio,
                   ratio_mean_std_dev_tokens, ratio_mean_std_dev_unique_tokens,
                   pca_noise_variance]

df_corpus_part01 = pd.DataFrame(value_variables).T
df_corpus_part01.columns = name_columns


# %% creating final df_corpus - part 02
df_corpus_part02 = pd.DataFrame(score_metric).T
df_corpus_part02.columns = name_metric


# %% creating final df_corpus - concat
df_corpus_final = pd.concat([df_corpus_part01, df_corpus_part02], axis=1)
df_corpus_final.to_csv(
    "sa_01_meta.csv", sep=";", index=False)


# %%
with open("processed_files.txt", "r") as f:
    document_file = f.read().split("\n")
selected_files = [
    file_name for file_name in document_file if file_name[6:15] != "explained"]
