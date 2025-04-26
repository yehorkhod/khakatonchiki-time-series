import pandas as pd
from preprocess import Preprocess, Config

# Read the data
df: pd.DataFrame = pd.read_parquet("data.parquet")

# Config
config: Config = Config(
    NaNs="Linear interpolation",
    Smoothing="Rolling median",
    Lags=True,
    Decays=True,
    PCA=True,
    test_size=0.2
)

# Preprocess
train, test = Preprocess(df, config).preprocess()
