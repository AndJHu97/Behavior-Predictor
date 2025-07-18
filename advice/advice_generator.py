import pandas as pd
  
#File headings: Category,Subcategory,Tags,Advice,Conditions
#i.e. Protective Role,Natural Tendencies,"guardian","You may rely on instinctive defense.","productive_drive < destructive_drive"
def load_advice_file(filepath):
    df = pd.read_csv(filepath)
    #Convert tags into an array rather than just one string separated by comma
    df["Tags"] = df["Tags"].apply(lambda x: [tag.strip() for tag in x.split(",") if x])
    return df

def evaluate_conditions(row, scores):
    condition = row.get("Conditions", "")
    if not condition or pd.isna(condition):
        return True  # No condition = always show
    try:
        return eval(condition, {}, scores)
    except Exception as e:
        print(f"⚠️ Condition Error '{condition}': {e}")
        return False

def filter_advice(advice_df, scores):
    filtered_rows = []

    for _, row in advice_df.iterrows():
        if evaluate_conditions(row, scores):
            filtered_rows.append(row)

    return pd.DataFrame(filtered_rows)

def export_advice(advice_df, filename="filtered_advice.csv", text_filename="filtered_advice.txt"):
    # Export CSV
    advice_df.to_csv(filename, index=False)
    print(f"✅ Exported {len(advice_df)} advice entries to {filename}")

    # Export readable text
    with open(text_filename, "w", encoding="utf-8") as f:
        grouped = advice_df.groupby(["Category", "Subcategory"])
        #Group is all the rows with the same category and subcategory
        for (category, subcategory), group in grouped:
            f.write(f"## {category} - {subcategory}\n\n")
            for _, row in group.iterrows():
                f.write(f"- {row['Advice']}\n")
            f.write("\n")
    
    print(f"📄 Also saved a readable version to {text_filename}")


'''
Example of how it'll be used
def main():
    # 🔢 Example user scores
    scores = {
        "productive_drive": 8,
        "destructive_drive": 5,
        "guardian": 7,
        "engaging_relationship": 4,
        "withdrawn_relationship": 7,
    }

    advice_df = load_advice_file("advice.csv")
    filtered_advice = filter_advice(advice_df, scores)
    export_advice(filtered_advice)
'''