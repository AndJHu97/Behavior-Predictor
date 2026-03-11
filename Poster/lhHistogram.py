import pandas as pd
import matplotlib.pyplot as plt

# --- Load the CSV ---
df = pd.read_csv("dbHigh_lh.csv")

# --- Choose the column ---
lh_column = "learned_helplessness"

# --- Plot histogram ---
plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(df[lh_column].dropna(), bins=45, color='skyblue', edgecolor='black')
plt.xlabel("Learned Helplessness Occurrences")
plt.ylabel("Number of Agents")
plt.title("Distribution of Learned Helplessness Occurrences")
plt.grid(axis='y', alpha=0.75)

# --- Optional: mark a tentative cutoff (you can change this after visual inspection) ---
# For example, if you see a gap around LH = 10
cutoff = 10
plt.axvline(cutoff, color='red', linestyle='--', linewidth=2, label=f"Cutoff = {cutoff}")
plt.legend()

plt.show()

# --- Optional: list agents below the cutoff ---
resilient_agents = df[df[lh_column] <= cutoff]
print(f"Number of resilient agents (LH ≤ {cutoff}): {len(resilient_agents)}")
