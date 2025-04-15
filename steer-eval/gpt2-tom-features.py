
import json


file_tom = "../feature-mining/tom_avg_sae_gpt2.json"
file_arc = "../feature-mining/avg_sae_gpt2.json"
file_out = "../feature-mining/arc-tom-gpt2.json"

with open(file_tom, "r") as f:
    tom = json.load(f)
with open(file_arc, "r") as f:
    arc = json.load(f)

if len(tom) != len(arc):
    raise ValueError(f"Size mismatch: tom has {len(tom)} elements, but arc has {len(arc)} elements.")

out = []
for i in range(len(tom)):
    diff = [t - a for t, a in zip(tom[i], arc[i])]
    out.append(diff)

with open(file_out, "w") as f:
    json.dump(out, f)

    top_activations = []

    # Original top activations code
    for layer in out:
        top_indices = sorted(range(len(layer)), key=lambda i: layer[i], reverse=True)[:5]
        top_values = [layer[i] for i in top_indices]
        top_activations.append(list(zip(top_indices, top_values)))

    for i, layer_activations in enumerate(top_activations):
        print(f"Layer {i + 1}:")
        for idx, value in layer_activations:
            print(f"  Dimension {idx}: Activation {value}")
    
    import matplotlib.pyplot as plt

    # Calculate means and variances for each layer
    means = [sum(layer) / len(layer) for layer in out]
    variances = [sum((x - mean) ** 2 for x in layer) / len(layer) for layer, mean in zip(out, means)]

    # Plot the mean and variance for each layer
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(means) + 1), means, label="Mean", marker="o")
    plt.plot(range(1, len(variances) + 1), variances, label="Variance", marker="x")
    plt.xlabel("Layer")
    plt.ylabel("Value")
    plt.title("Mean and Variance of Activations per Layer")
    plt.legend()
    plt.grid(True)
    plt.show()


