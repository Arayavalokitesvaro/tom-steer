# Let's redefine everything for two model families: gpt2 and pythia
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tabulate import tabulate

# Define base directory and model groups
base_dir = "../feature-mining/output"
model_groups = {
    "gpt2": {
        "tom_prefix": "gpt2_avg_sae_layer",
        "base_prefix": "gpt2_avg_base_layer"
    },
    "pythia": {
        "tom_prefix": "pythia-70m-deduped_avg_sae_layer",
        "base_prefix": "pythia-70m-deduped_avg_base_layer"
    }
}

# Function to load and compute difference
def compute_diff_and_stats(model_name, tom_prefix, base_prefix):
    tom_files = sorted([f for f in os.listdir(base_dir) if f.startswith(tom_prefix) and "5994" not in f])
    base_files = sorted([f for f in os.listdir(base_dir) if f.startswith(base_prefix) and "5994" not in f])

    if len(tom_files) != len(base_files):
        raise ValueError(f"[{model_name}] Mismatch between number of ToM and base files")

    diffs, tops, means, variances, max_vals, min_vals, q1s, q3s, medians = [], [], [], [], [], [], [], [], []

    for tom_file, base_file in zip(tom_files, base_files):
        with open(os.path.join(base_dir, tom_file), "r") as f:
            tom_data = json.load(f)
        with open(os.path.join(base_dir, base_file), "r") as f:
            base_data = json.load(f)

        if isinstance(tom_data[0], list):
            layer_diff = [[t - b for t, b in zip(t_row, b_row)] for t_row, b_row in zip(tom_data, base_data)]
        else:
            layer_diff = [t - b for t, b in zip(tom_data, base_data)]

        diffs.append(layer_diff)

        flat_layer = sum(layer_diff, []) if isinstance(layer_diff[0], list) else layer_diff
        print("Befor zeroing ", len(flat_layer))
        flat_layer = [x for x in flat_layer if not np.isclose(x, 0, atol=1e-5)]
        print("After zeroing ", len(flat_layer))
        means.append(sum(flat_layer) / len(flat_layer))
        variances.append(sum((x - means[-1]) ** 2 for x in flat_layer) / len(flat_layer))
        max_vals.append(max(flat_layer))
        min_vals.append(min(flat_layer))
        q1s.append(np.percentile(flat_layer, 25))
        q3s.append(np.percentile(flat_layer, 75))
        medians.append(np.median(flat_layer))
        top_indices = sorted(range(len(flat_layer)), key=lambda i: flat_layer[i], reverse=True)[:5]
        top_values = [flat_layer[i] for i in top_indices]
        tops.append(list(zip(top_indices, top_values)))

    return {
        "diffs": diffs,
        "tops": tops,
        "means": means,
        "variances": variances,
        "max_vals": max_vals,
        "min_vals": min_vals,
        "q1s": q1s,
        "q3s": q3s,
        "medians": medians
    }

# Process both model groups
results = {}
for model_name, prefixes in model_groups.items():
    results[model_name] = compute_diff_and_stats(
        model_name,
        prefixes["tom_prefix"],
        prefixes["base_prefix"]
    )

# Plotting means, variances, max activations, and min activations
plt.figure(figsize=(12, 6))
for model_name in results:
    plt.plot(results[model_name]["means"], label=f"{model_name} Mean", marker="o")
    # plt.plot(results[model_name]["variances"], label=f"{model_name} Variance", marker="x")
    plt.plot(results[model_name]["max_vals"], label=f"{model_name} Max", marker="^")
    plt.plot(results[model_name]["min_vals"], label=f"{model_name} Min", marker="v")

plt.xlabel("Layer")
plt.ylabel("Activation Difference")
plt.title("ToM v.s. Non-ToM SAE Activation Difference per Layer: Max, and Min")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# Extract one model

# Choose the first model
model_name = list(results.keys())[0]
data = results[model_name]

layers = np.arange(len(data["medians"]))
medians = np.array(data["medians"])
q1s = np.array(data["q1s"])
q3s = np.array(data["q3s"])
mins = np.array(data["min_vals"])
maxs = np.array(data["max_vals"])

# Compute IQR as asymmetric error bars
lower_iqr = np.clip(medians - q1s, 0, None)
upper_iqr = np.clip(q3s - medians, 0, None)
iqr_error = [lower_iqr, upper_iqr]

plt.figure(figsize=(12, 6))

# Plot median with IQR error bars
plt.errorbar(
    layers,
    medians,
    yerr=iqr_error,
    fmt='-o',
    capsize=5,
    label="Median ± IQR",
    color="blue"
)

# Plot min and max as dashed lines
plt.plot(layers, mins, linestyle="--", marker="v", label="Min", color="gray")
plt.plot(layers, maxs, linestyle="--", marker="^", label="Max", color="gray")

plt.xlabel("Layer")
plt.ylabel("Activation Difference")
plt.title(f"{model_name}: Median ± IQR with Min/Max")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




# Combine top activations into a readable table
top_df = pd.DataFrame(columns=["Model", "Layer", "Top Features"])
for model_name, data in results.items():
    for i, layer in enumerate(data["tops"]):
        top_df = pd.concat([
            top_df,
            pd.DataFrame([{
                "Model": model_name,
                "Layer": i,
                "Top Features": layer
            }])
        ], ignore_index=True)

print(tabulate(top_df, headers="keys", tablefmt="pretty"))

# Save the top_df into a CSV file
output_csv_path = os.path.join(base_dir, "top_activations.csv")
top_df.to_csv(output_csv_path, index=False)
print(f"Top activations saved to {output_csv_path}")