from datasets import load_dataset
import pandas as pd
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from sae_lens.sae import SAE
import json
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load ARC-Easy dataset (first 10 examples for testing)
ds = load_dataset("allenai/ai2_arc", "ARC-Easy")["train"]
df = pd.DataFrame(ds)

# Model and SAE configs
models = ['EleutherAI/pythia-70m-deduped', 'google/gemma-2-2b']
saes = ['pythia-70m-deduped-res-sm', 'gemma-scope-2b-pt-res']

# Define the number of layers for each model
n_layers = {
    'gemma-scope-2b-pt-res': 26,  # As per the SAE table
    'pythia-70m-deduped-res-sm': 6  # As per the SAE table
}

# Function to get the hook name based on SAE and layer
def get_hook_name(sae_name, layer):
    if sae_name == 'gemma-scope-2b-pt-res':
        return f"blocks.{layer}.hook_resid_post"
    elif sae_name == 'pythia-70m-deduped-res-sm':
        return f"blocks.{layer}.hook_mlp_out"
    else:
        raise ValueError(f"Unknown SAE name: {sae_name}")


for i in range(len(models)):
    print(f"\n==== Evaluating Model: {models[i]} ====")

    # Load model
    model = HookedTransformer.from_pretrained(models[i], device=device)
    tokenizer = model.tokenizer
    num_layers = model.cfg.n_layers
    model.eval()

    # Load SAEs
    this_sae = []
    for layer in range(num_layers):
        sae, _, sparsity = SAE.from_pretrained(
            release=saes[i],
            sae_id=f"blocks.{layer}.hook_resid_post",
            device=device,
        )
        this_sae.append(sae)
        print(f"Layer {layer}: SAE loaded with sparsity {sparsity.mean().item():.2f}")

    # Online mean accumulators (sum and token counts)
    layer_running_sum = [torch.zeros(sae.cfg.d_sae, device=device) for sae in this_sae]
    layer_token_count = [0 for _ in range(num_layers)]

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]
        choices = row["choices"]["text"]
        labels = row["choices"]["label"]
        answer_key = row["answerKey"]

        logprobs = []

        for choice in choices:
            prompt = f"{question} {choice}"
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                logits, cache = model.run_with_cache(prompt)
                logits = logits[:, :-1, :]
                probs = torch.nn.functional.log_softmax(logits, dim=-1)
                target_ids = input_ids[:, 1:]
                choice_logprob = torch.gather(probs, 2, target_ids.unsqueeze(-1)).sum().item()
                logprobs.append(choice_logprob)

                # SAE activation across all tokens → accumulate globally
                for l in range(num_layers):
                    resid_pre_all = cache[f"blocks.{l}.hook_resid_pre"]  # shape: (seq_len, d_model)
                    encoded = this_sae[l].encode(resid_pre_all.to(device))  # shape: (1, seq_len, d_sae)
                    encoded = encoded.squeeze(0)  # shape: (seq_len, d_sae)

                    if encoded.shape[1] != layer_running_sum[l].shape[0]:
                        raise ValueError(f"[Layer {l}] Final encoded shape: {encoded.shape}, expected d_sae={layer_running_sum[l].shape[0]}")

                    layer_running_sum[l] += encoded.sum(dim=0)
                    layer_token_count[l] += encoded.shape[0]

        pred_idx = int(torch.tensor(logprobs).argmax())
        correct = labels[pred_idx] == answer_key

        results.append({
            "id": row["id"],
            "question": question,
            "choices": choices,
            "labels": labels,
            "answer_key": answer_key,
            "predicted": labels[pred_idx],
            "correct": correct,
            "logprobs": [float(lp) for lp in logprobs]
        })

    # Accuracy report
    accuracy = sum(r["correct"] for r in results) / len(results)
    print(f"✅ Model Accuracy on ARC-Easy: {accuracy:.2%}")

    # Save prediction results
    model_name = models[i].split("/")[-1]
    with open(f"results_{model_name}.pkl", "wb") as f:
        pickle.dump(results, f)

    # Compute and save average SAE activations per layer
    avg_sae = [
        (layer_sum / layer_token_count[l]).detach().cpu().numpy().tolist()
        for l, layer_sum in enumerate(layer_running_sum)
    ]
    with open(f"avg_sae_{model_name}.json", "w") as f:
        json.dump(avg_sae, f)

    print(f"💾 Saved averaged SAE activations to avg_sae_{model_name}.json")
    torch.cuda.empty_cache()
