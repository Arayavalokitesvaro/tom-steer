import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

df = pd.read_csv("./steer-list.csv")
df["accuracy"] = None

data = []
for _, row in df.iterrows():
    model = row["model"]
    layer = row["layer"]
    feature = row["feature"]
    for strength in [5.0, 10.0, 15.0]:
        # Run evaluation
        filename = f"./output/accuracy_{model}_{layer}_index{feature}_strength{strength}.txt"
        if not os.path.exists(filename):
            print(f"Output file {filename} already exists. Skipping...")
            continue
        accuracy = None
        with open(filename, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "Steering Accuracy" in line:
                    accuracy = float(line.split(":")[1].split("(")[0].split("%")[0].strip()) * 0.01
                    break
            else:
                print(f"No accuracy found in {filename}")
                continue
        data.append({
            "model": model,
            "layer": layer,
            "feature": feature,
            "strength": strength,
            "accuracy": accuracy
        })

# Convert data to DataFrame
plot_df = pd.DataFrame(data)

# Filter for gpt2-small model
plot_df = plot_df[plot_df["model"] == "gpt2-small"]

# Plot each line
plt.figure(figsize=(10, 6))
for (layer, feature), group in plot_df.groupby(["layer", "feature"]):
    plt.plot(group["strength"], group["accuracy"], marker='o', label=f"Layer {layer}, Feature {feature}")

# Add baseline accuracy
plt.axhline(y=0.485, color='r', linestyle='--', label="Baseline Accuracy (48.5%)")

# Customize plot
plt.title("Steering Accuracy vs Strength for gpt2-small")
plt.xlabel("Strength")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)

# Show plot
plt.show()
