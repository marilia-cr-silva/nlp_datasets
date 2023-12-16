# %%
import matplotlib.pyplot as plt
import numpy as np
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
    0.9594,
    0.5405,
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
    0.8780,
    0.4935,
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
    0.8154,
    0.5394,
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
    0.7922,
    0.6565,
]

# %%
datasets = [f"hs_{i:02d}" for i in range(1, 16)]


# %%
def plot(df: pd.DataFrame, title: str, rgb_hex: str) -> None:
    sns.set()
    sns.set_style("whitegrid")
    ax = sns.barplot(data=df, color=rgb_hex, ci=None)  # , width=0.8)
    ax.set_title(title)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_xticklabels(rotation=45, labels=datasets)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim([0.0, 1.0])
    plt.show()


# %% Plotting AutoGluon
df_autogluon = pd.DataFrame(dict(zip(datasets, list_autogluon)), index=range(1))
plot(df_autogluon, title="AutoGluon", rgb_hex="#66c2a5")

# %% Plotting auto-sklearn
df_autosklearn = pd.DataFrame(dict(zip(datasets, list_autosklearn)), index=range(1))
plot(df_autosklearn, title="auto-sklearn", rgb_hex="#fc8d62")

# %% Plotting Logistic Regression
df_logistic_regression = pd.DataFrame(
    dict(zip(datasets, list_logistic_regression)), index=range(1)
)
plot(df_logistic_regression, title="Logistic Regression", rgb_hex="#8da0cb")

# %% Plotting TPOT
df_tpot = pd.DataFrame(dict(zip(datasets, list_TPOT)), index=range(1))
plot(df_tpot, title="TPOT", rgb_hex="#a6d854")

# %%
df = pd.DataFrame(
    {
        "autogluon": list_autogluon,
        "logistic_regression": list_logistic_regression,
        "tpot": list_TPOT,
        "autosklearn": list_autosklearn,
    }
)

# %%
X = [f"hs_{i:02d}" for i in range(1, 16)]
X_axis = np.arange(len(X))
plt.bar(X_axis - 0.4, list_autogluon, 0.2, hatch="---", label="AutoGluon")
plt.bar(X_axis - 0.2, list_autosklearn, 0.2, label="Auto-Sklearn", hatch="\\\\")
plt.bar(X_axis + 0.0, list_logistic_regression, 0.2, label="LR+RS", hatch="//")
plt.bar(X_axis + 0.2, list_TPOT, 0.2, label="TPOT", hatch="..")

plt.xticks(X_axis, X, rotation=45)
# plt.xlabel("Datasets")
plt.ylabel("Balanced Accuracy")
# plt.title("Balanced Accuracy per Dataset")
plt.grid(False)
plt.legend()
plt.show()
