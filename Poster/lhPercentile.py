import pandas as pd
import numpy as np

# --- Load the CSV ---
df = pd.read_csv("dbHigh_lh.csv")

# --- Choose the column ---
lh_column = "learned_helplessness"

# --- Choose the percentile for "resilience" cutoff ---
percentile_value = 25  # e.g., bottom 25% are considered resilient

# --- Calculate the cutoff ---
cutoff = np.percentile(df[lh_column].dropna(), percentile_value)
print(f"Resilience cutoff at {percentile_value}th percentile: {cutoff}")

# --- Identify resilient agents ---
# Assuming each row represents one agent
resilient_agents = df[df[lh_column] <= cutoff]
print(f"Number of resilient agents: {len(resilient_agents)}")
print("Resilient agents:")
print(resilient_agents)

# --- Optional: save the resilient agents to a CSV ---
#resilient_agents.to_csv("resilient_agents.csv", index=False)
#print("Resilient agents saved to 'resilient_agents.csv'")
