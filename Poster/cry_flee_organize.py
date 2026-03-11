import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv("cryfleeLow.csv")  # columns: run_id, cry, flee

# Compute raw difference (for inspection)
df["d"] = df["cry"] - df["flee"]

# Compute normalized dominance ratio in [-1, 1]
df["dominance_ratio"] = (
    (df["cry"] - df["flee"]) /
    (df["cry"] + df["flee"] + 1e-8)
)

# Set dominance threshold
delta = 0.10  # 10% relative dominance
print(f"Using dominance ratio delta: {delta}")

# Initialize columns
df["cry_dominant"] = np.nan
df["flee_dominant"] = np.nan
df["dominance_state"] = "indeterminate"

# Define dominance masks
cry_mask = df["dominance_ratio"] > delta
flee_mask = df["dominance_ratio"] < -delta

# Apply dominance rules
df.loc[cry_mask, "cry_dominant"] = df.loc[cry_mask, "cry"]
df.loc[flee_mask, "flee_dominant"] = df.loc[flee_mask, "flee"]

df.loc[cry_mask, "dominance_state"] = "cry_dominant"
df.loc[flee_mask, "dominance_state"] = "flee_dominant"

# Save output
df.to_csv("cryfleeLow_dominance.csv", index=False)
