import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
import os

# Load params
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Load & clean data
df = pd.read_csv("data/raw/telco_churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Split data
train, test = train_test_split(
    df,
    test_size=params["preprocess"]["test_size"],
    random_state=params["preprocess"]["random_state"]
)

# Save processed data
os.makedirs("data/processed", exist_ok=True)
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)