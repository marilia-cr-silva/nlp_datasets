# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %%
list_autogluon = [
    0.8082,
    0.6692,
    0.8009,
    0.6114,
    0.5029,
    0.6025,
    0.6917,
    0.8121,
    0.5000,
    0.7539,
    0.8650,
    0.7208,
    0.4461,
]
list_autosklearn = [
    0.7863,
    0.6265,
    0.8133,
    0.5000,
    0.3864,
    0.6189,
    0.6478,
    0.7663,
    0.5000,
    0.7545,
    0.8517,
    0.6494,
    0.4415,
]
list_logistic_regression = [
    0.7622,
    0.6692,
    0.8134,
    0.7998,
    0.5327,
    0.6230,
    0.6867,
    0.7571,
    0.4960,
    0.7593,
    0.8608,
    0.7056,
    0.4702,
]
list_TPOT = [
    0.7993,
    0.7325,
    0.7923,
    0.8147,
    0.5838,
    0.5930,
    0.6862,
    0.8481,
    0.5038,
    0.7360,
    0.8441,
    0.6732,
    0.6244,
]

# %%
datasets = [f"hs_{i:02d}" for i in range(1, 14)]


# %%
def plot(df: pd.DataFrame, title: str, rgb_hex: str) -> None:
    sns.set()
    sns.set_style("whitegrid")
    ax = sns.barplot(data=df, color=rgb_hex, ci=None, width=0.8)
    ax.set_title(title)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_xticklabels(rotation=45, labels=datasets)
    plt.show()


# %% Plotting AutoGluon
df_autogluon = pd.DataFrame(
    dict(zip(datasets, list_autogluon)), index=range(1))
plot(df_autogluon, title="AutoGluon", rgb_hex="#66c2a5")

# %% Plotting auto-sklearn
df_autosklearn = pd.DataFrame(
    dict(zip(datasets, list_autosklearn)), index=range(1))
plot(df_autosklearn, title="auto-sklearn", rgb_hex="#fc8d62")

# %% Plotting Logistic Regression
df_logistic_regression = pd.DataFrame(
    dict(zip(datasets, list_logistic_regression)), index=range(1)
)
plot(df_logistic_regression, title="Logistic Regression", rgb_hex="#8da0cb")

# %% Plotting TPOT
df_tpot = pd.DataFrame(dict(zip(datasets, list_TPOT)), index=range(1))
plot(df_tpot, title="TPOT", rgb_hex="#a6d854")
