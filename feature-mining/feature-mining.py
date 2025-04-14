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

    # Get model predictions and cache all logits
    results = []
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

    # Run SAE one layer at a time and save immediately
    for layer in range(num_layers):
        print(f"\n🔄 Processing SAE layer {layer}")

        sae, _, _ = SAE.from_pretrained(
            release=saes[i],
            sae_id=get_sae_id(saes[i], layer),
            device=device
        )
        layer_running_sum = torch.zeros(sae.cfg.d_sae, device=device)
        layer_token_count = 0

        for _, row in tqdm(df.iterrows(), total=len(df)):
            question = row["question"]
            choices = row["choices"]["text"]

            for choice in choices:
                prompt = f"{question} {choice}"
                with torch.no_grad():
                    _, cache = model.run_with_cache(prompt)
                    resid_pre = cache[f"blocks.{layer}.hook_resid_pre"]
                    encoded = sae.encode(resid_pre.to(device)).squeeze(0)
                    if encoded.shape[1] != layer_running_sum.shape[0]:
                        raise ValueError(f"[Layer {layer}] shape mismatch: got {encoded.shape}, expected {layer_running_sum.shape[0]}")
                    layer_running_sum += encoded.sum(dim=0)
                    layer_token_count += encoded.shape[0]

                del cache, resid_pre, encoded, prompt
                torch.cuda.empty_cache()

        # Save average SAE activation for this layer
        avg_sae = (layer_running_sum / layer_token_count).detach().cpu().numpy().tolist()
        model_name = models[i].split("/")[-1]
        with open(f"avg_sae_{model_name}_layer{layer}.json", "w") as f:
            json.dump(avg_sae, f)
        print(f"💾 Saved avg SAE activations for layer {layer} to avg_sae_{model_name}_layer{layer}.json")

        del sae
        torch.cuda.empty_cache()

    # Final accuracy
    accuracy = sum(r["correct"] for r in results) / len(results)
    print(f"✅ Model Accuracy on ARC-Easy: {accuracy:.2%}")

    model_name = models[i].split("/")[-1]
    with open(f"results_{model_name}.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"💾 Saved prediction results to results_{model_name}.pkl")

    torch.cuda.empty_cache()
