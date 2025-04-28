file_path = "../feature-mining/output/top_activations.csv"

import pandas as pd
import os
# import saelens

df = pd.read_csv(file_path)
df["Explanations"] = None
import requests
from tabulate import tabulate

modelIds = ["gpt2-small", "pythia-70m-deduped"]
sadIds = ["-res-jb", "-res-sm"]
for _, row in df.iterrows():
    modelId= "gpt2-small" if row["Model"] == "gpt2" else "pythia-70m-deduped"
    saeId = "-res-jb" if row["Model"] == "gpt2" else "-res-sm"

    url = f"https://www.neuronpedia.org/api/explanation/export?modelId={modelId}&saeId={row['Layer']}{saeId}"
    print("Reading from ", url)
    headers = {"Content-Type": "application/json"}

    response = requests.get(url, headers=headers)
    # convert to pandas
    
    if response.status_code == 200:
        data = response.json()
    else:
        print(f"Failed to fetch data from {url}. Status code: {response.status_code}")
        continue
    explanations_df = pd.DataFrame(data)
    # rename index to "feature"
    explanations_df.rename(columns={"index": "feature"}, inplace=True)
    # explanations_df["feature"] = explanations_df["feature"].astype(int)
    explanations_df["description"] = explanations_df["description"].apply(
        lambda x: x.lower()
    )
    top_features = eval(row["Top Features"])  # Convert string representation to list of tuples
    explanations = []
    for idx, _ in top_features:
        explanation = explanations_df[
            (explanations_df["feature"] == idx) & 
            (explanations_df["explanationModelName"] == "gpt-3.5-turbo")
        ]["description"].values
        if len(explanation):
            explanations.append(explanation[0])
        else:
            explanations.append(None)  # Handle case where no explanation is found
    row["Explanations"] = explanations

# Save the updated DataFrame back to the original CSV
df.to_csv(file_path, index=False)
# Print the DataFrame as a table
print(tabulate(df, headers='keys', tablefmt='grid'))