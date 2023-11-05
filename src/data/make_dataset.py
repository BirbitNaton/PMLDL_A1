import pandas as pd
from pandarallel import pandarallel
from pathlib import Path

pandarallel.initialize(progress_bar=True)

data_dir = Path("data")
dataset_path = data_dir / "raw/filtered.tsv"
formatted_dataset_path = data_dir / "interim/formatted.parquet"

df = pd.read_csv(dataset_path, sep="\t")

# rearrange ref and trn so that ref_tox >= trn_tox
trn_tox_cond = df["trn_tox"] > df["ref_tox"]
df.loc[trn_tox_cond, ["reference", "translation", "ref_tox", "trn_tox"]] = df.loc[trn_tox_cond, ["translation", "reference", "trn_tox", "ref_tox"]].values

# calc fit_score
df["tox_diff"] = df["ref_tox"]-df["trn_tox"]
df["fit_score"] = df["tox_diff"] * df["similarity"]
df = df.sort_values(by="fit_score", ascending=False).reset_index(drop=True)

# outliers removal
trn_tox_p = df["trn_tox"].quantile(0.9)
ref_tox_p = df["ref_tox"].quantile(0.1)
quantile = df["fit_score"].quantile(0.1)

df = df[df["trn_tox"] < trn_tox_p]
df = df[df["ref_tox"] > ref_tox_p]
df = df[df["fit_score"] > quantile].reset_index(drop=True)

df.reset_index(drop=True).to_parquet(formatted_dataset_path, index=None)