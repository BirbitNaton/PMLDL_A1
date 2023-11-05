import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns

figures_dir = Path("reports/figures")
data_dir = Path("data")

dataset_path = data_dir / "raw/filtered.tsv"

# load data
df = pd.read_csv(dataset_path, sep="\t")

# visualize stuff
cols = ["score", "label"]
ref_tox = pd.concat([df["ref_tox"], pd.Series("ref_tox", index=df["ref_tox"].index)], axis=1)
trn_tox = pd.concat([df["trn_tox"], pd.Series("trn_tox", index=df["trn_tox"].index)], axis=1)
ref_tox.columns, trn_tox.columns = cols, cols

tox_df = pd.concat([ref_tox, trn_tox])

plt.rcParams['figure.dpi'] = 180
plt.rcParams['savefig.dpi'] = 180
plot = sns.displot(data=tox_df, x="score", hue="label", palette="autumn", alpha=0.5, linewidth=0.3)
title = "Toxicity Scores Distribution"
plt.title(title)

plot.savefig(figures_dir / (title + ".png"))

# rearrange ref and trn so that ref_tox >= trn_tox
trn_tox_cond = df["trn_tox"] > df["ref_tox"]
df.loc[trn_tox_cond, ["reference", "translation", "ref_tox", "trn_tox"]] = df.loc[trn_tox_cond, ["translation", "reference", "trn_tox", "ref_tox"]].values

# visualize more stuff
cols = ["score", "label"]
ref_tox = pd.concat([df["ref_tox"], pd.Series("ref_tox", index=df["ref_tox"].index)], axis=1)
trn_tox = pd.concat([df["trn_tox"], pd.Series("trn_tox", index=df["trn_tox"].index)], axis=1)
ref_tox.columns, trn_tox.columns = cols, cols

tox_df = pd.concat([ref_tox, trn_tox])

plt.rcParams['figure.dpi'] = 180
plt.rcParams['savefig.dpi'] = 180
plot = sns.displot(data=tox_df, x="score", hue="label", palette="autumn", alpha=0.5, linewidth=0.3)

bbox = dict(facecolor='green', alpha=0.3, pad=0.05, edgecolor='none')
trn_tox_p = df["trn_tox"].quantile(0.9)
plt.axvline(trn_tox_p, 0, 1, color="black", linewidth=0.75)
plt.text(trn_tox_p, -1e4, trn_tox_p.round(5), fontsize=8, bbox=bbox)

bbox = dict(facecolor='red', alpha=0.3, pad=0.05, edgecolor='none')
ref_tox_p = df["ref_tox"].quantile(0.1)
plt.axvline(ref_tox_p, 0, 1, color="black", linewidth=0.75)
plt.text(ref_tox_p, -1e4, ref_tox_p.round(5), fontsize=8, bbox=bbox)
title = "Polarized Toxicity Scores Distribution [mark(0.1) and mark(0.9) quantiles]"
plt.title(title)

plot.savefig(figures_dir / (title + ".png"))


# visualize more stuff
df = df[df["trn_tox"] < trn_tox_p]

cols = ["score", "label"]
ref_tox = pd.concat([df["ref_tox"], pd.Series("ref_tox", index=df["ref_tox"].index)], axis=1)
trn_tox = pd.concat([df["trn_tox"], pd.Series("trn_tox", index=df["trn_tox"].index)], axis=1)
ref_tox.columns, trn_tox.columns = cols, cols

tox_df = pd.concat([ref_tox, trn_tox])

plt.rcParams['figure.dpi'] = 180
plt.rcParams['savefig.dpi'] = 180
plot = sns.displot(data=tox_df, x="score", hue="label", palette="autumn", alpha=0.5, linewidth=0.3)

bbox = dict(facecolor='green', alpha=0.3, pad=0.05, edgecolor='none')
# As we have already adjusted this border, let's leave it's value as is 
plt.axvline(trn_tox_p, 0, 1, color="black", linewidth=0.75)
plt.text(trn_tox_p, -1e4, trn_tox_p.round(5), fontsize=8, bbox=bbox)

bbox = dict(facecolor='red', alpha=0.3, pad=0.05, edgecolor='none')
ref_tox_p = df["ref_tox"].quantile(0.1)
plt.axvline(ref_tox_p, 0, 1, color="black", linewidth=0.75)
plt.text(ref_tox_p, -1e4, ref_tox_p.round(5), fontsize=8, bbox=bbox)
title = "Polarized Toxicity Scores Distribution [cut(0.1) and mark(0.9) quantiles]"
plt.title(title)

plot.savefig(figures_dir / (title + ".png"))


# visualize more stuff
df = df[df["ref_tox"] > ref_tox_p]

cols = ["score", "label"]
ref_tox = pd.concat([df["ref_tox"], pd.Series("ref_tox", index=df["ref_tox"].index)], axis=1)
trn_tox = pd.concat([df["trn_tox"], pd.Series("trn_tox", index=df["trn_tox"].index)], axis=1)
ref_tox.columns, trn_tox.columns = cols, cols

tox_df = pd.concat([ref_tox, trn_tox])

plt.rcParams['figure.dpi'] = 180
plt.rcParams['savefig.dpi'] = 180
plot = sns.displot(data=tox_df, x="score", hue="label", palette="autumn", alpha=0.5, linewidth=0.3)

bbox = dict(facecolor='green', alpha=0.3, pad=0.05, edgecolor='none')
# As we have already adjusted this border, let's leave it's value as is 
plt.axvline(trn_tox_p, 0, 1, color="black", linewidth=0.75)
plt.text(trn_tox_p, -1e4, trn_tox_p.round(5), fontsize=8, bbox=bbox)

bbox = dict(facecolor='red', alpha=0.3, pad=0.05, edgecolor='none')
# As we have already adjusted this border, let's leave it's value as is 
plt.axvline(ref_tox_p, 0, 1, color="black", linewidth=0.75)
plt.text(ref_tox_p, -1e4, ref_tox_p.round(5), fontsize=8, bbox=bbox)
title = "Polarized Toxicity Scores Distribution [cut(0.1) and cut(0.9) quantiles]"
plt.title(title)

plot.savefig(figures_dir / (title + ".png"))


# visualize more stuff
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200
plot = sns.displot(df["ref_tox"] - df["trn_tox"], linewidth=0.1, color="darkorange")

min_diff = (df["ref_tox"] - df["trn_tox"]).min()
plt.axvline(min_diff, 0, 1, color="black", linewidth=0.75)
plt.text(min_diff, -1.25e3, min_diff.round(2), fontsize=8, bbox=bbox)
title = "Difference of Reference and Translated Scores Distribution [marked(min)]"
plt.title(title)

plot.savefig(figures_dir / (title + ".png"))

# make fit_score metric
df["tox_diff"] = df["ref_tox"]-df["trn_tox"]
df["fit_score"] = df["tox_diff"] * df["similarity"]
df = df.sort_values(by="fit_score", ascending=False).reset_index(drop=True)


# visualize more stuff
plt.clf()
plot = sns.heatmap(df[["tox_diff", "trn_tox", "ref_tox", "fit_score", "similarity"]].corr())
title = "Correlation matrix"
plt.title(title)

plot.figure.savefig(figures_dir / (title + ".png"))


# visualize more stuff
plt.clf()
plot = sns.lineplot(data=df.sample(frac=0.01), x="similarity", y="tox_diff", linewidth=0.1, color="darkorange")
title = "Similarity-tox_diff dependancy"
plt.title(title)

plot.figure.savefig(figures_dir / (title + ".png"))


# visualize more stuff
plt.clf()
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200

plot = sns.displot(df["fit_score"], color="darkorange")
quantile = df["fit_score"].quantile(0.1)
min_score = df["fit_score"].min()

bbox = dict(facecolor='green', alpha=0.3, pad=0.05, edgecolor='none')
plt.axvline(quantile, 0, 1, color="green", linewidth=0.75)
plt.text(quantile, -1.25e2, quantile.round(2), fontsize=8, bbox=bbox)

bbox = dict(facecolor='red', alpha=0.3, pad=0.05, edgecolor='none')
plt.axvline(min_score, 0, 1, color="red", linewidth=0.75)
plt.text(min_score, -1.25e2, min_score.round(2), fontsize=8, bbox=bbox)

title = "Fit Score Distribution"
plt.title(title)

plot.figure.savefig(figures_dir / (title + ".png"))