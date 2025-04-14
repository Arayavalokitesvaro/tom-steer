from datasets import load_dataset
import pandas as pd
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from sae_lens.sae import SAE
import json
import pickle
import os

# Memory fragmentation fix
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load ARC-Easy dataset
ds = load_dataset("allenai/ai2_arc", "ARC-Easy")["train"]
df = pd.DataFrame(ds)

# Model and SAE configs
models = ['google/gemma-2-2b']
saes = ['gemma-scope-2b-pt-res-canonical']

n_layers = {'gemma-scope-2b-pt-res': 26}

def get_sae_id(sae_name, layer):
    if sae_name == 'gemma-scope-2b-pt-res-canonical':
        return f"layer_{layer}/width_16k/canonical"
    elif sae_name == 'pythia-70m-deduped-res-sm':
        return f"blocks.{layer}.hook_resid_post"
    elif sae_name == 'gpt2-small-res-jb':
        return f"blocks.{layer}.hook_resid_pre"
    else:
        raise ValueError(f"Unknown SAE name: {sae_name}")

for i in range(len(models)):
    print(f"\n==== Evaluating Model: {models[i]} ====")

    model = HookedTransformer.from_pretrained(models[i], device=device)
    tokenizer = model.tokenizer
    num_layers = model.cfg.n_layers
    model.eval()
    torch.cuda.empty_cache()

    # Prepare result buffer and SAE accumulator
    layer_running_sum = [None for _ in range(num_layers)]
    layer_token_count = [0 for _ in range(num_layers)]
    results = []

    # Get model predictions and cache all logits
    for _, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]
        choices = row["choices"]["text"]
        labels = row["choices"]["label"]
        answer_key = row["answerKey"]

        logprobs = []

        for choice in choices:
            prompt = f"{question} {choice}"
            input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)

            with torch.no_grad():
                logits, cache = model.run_with_cache(prompt)
                logits = logits[:, :-1, :]
                probs = torch.nn.functional.log_softmax(logits, dim=-1)
                target_ids = input_ids[:, 1:]
                choice_logprob = torch.gather(probs, 2, target_ids.unsqueeze(-1)).sum().item()
                logprobs.append(choice_logprob)

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

    torch.cuda.empty_cache()

    # Chunked SAE evaluation (4 SAEs at a time)
    sae_chunk_size = 1
    num_chunks = (num_layers + sae_chunk_size - 1) // sae_chunk_size

    for chunk_id in range(num_chunks):
        start_layer = chunk_id * sae_chunk_size
        end_layer = min((chunk_id + 1) * sae_chunk_size, num_layers)
        print(f"\n🔄 Processing SAE layers {start_layer} to {end_layer - 1}")

        # Load chunk SAEs
        chunk_sae = {}
        for layer in range(start_layer, end_layer):
            sae, _, _ = SAE.from_pretrained(
                release=saes[i],
                sae_id=get_sae_id(saes[i], layer),
                device=device
            )
            chunk_sae[layer] = sae
            torch.cuda.empty_cache()

        # Initialize accumulators
        for l in range(start_layer, end_layer):
            layer_running_sum[l] = torch.zeros(chunk_sae[l].cfg.d_sae, device=device)
            layer_token_count[l] = 0

        # Pass all examples again
        for _, row in tqdm(df.iterrows(), total=len(df)):
            question = row["question"]
            choices = row["choices"]["text"]

            for choice in choices:
                prompt = f"{question} {choice}"
                with torch.no_grad():
                    _, cache = model.run_with_cache(prompt)

                    for l in range(start_layer, end_layer):
                        resid_pre = cache[f"blocks.{l}.hook_resid_pre"]
                        encoded = chunk_sae[l].encode(resid_pre.to(device)).squeeze(0)
                        if encoded.shape[1] != layer_running_sum[l].shape[0]:
                            raise ValueError(f"[Layer {l}] shape mismatch: got {encoded.shape}, expected {layer_running_sum[l].shape[0]}")
                        layer_running_sum[l] += encoded.sum(dim=0)
                        layer_token_count[l] += encoded.shape[0]

            torch.cuda.empty_cache()

        del chunk_sae
        torch.cuda.empty_cache()

    # Final accuracy
    accuracy = sum(r["correct"] for r in results) / len(results)
    print(f"✅ Model Accuracy on ARC-Easy: {accuracy:.2%}")

    # Save prediction results
    model_name = models[i].split("/")[-1]
    with open(f"results_{model_name}.pkl", "wb") as f:
        pickle.dump(results, f)

    # Save average SAE activations
    avg_sae = [
        (layer_running_sum[l] / layer_token_count[l]).detach().cpu().numpy().tolist()
        for l in range(num_layers)
    ]
    with open(f"avg_sae_{model_name}.json", "w") as f:
        json.dump(avg_sae, f)

    print(f"💾 Saved averaged SAE activations to avg_sae_{model_name}.json")
    torch.cuda.empty_cache()
