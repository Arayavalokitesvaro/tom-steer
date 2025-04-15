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
models = ['gpt2', 'google/gemma-2-2b', 'EleutherAI/pythia-70m-deduped']
saes = ['gpt2-small-res-jb' ,'gemma-scope-2b-pt-res-canonical', 'pythia-70m-deduped-res-sm']

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

    # Preload SAEs for each layer
    saes_by_layer = {}
    for layer in range(num_layers):
        sae, _, _ = SAE.from_pretrained(
            release=saes[i],
            sae_id=get_sae_id(saes[i], layer),
            device=device
        )
        saes_by_layer[layer] = {
            "sae": sae,
            "running_sum": torch.zeros(sae.cfg.d_sae, device=device),
            "token_count": 0
        }

    # Get model predictions and cache activations
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

                # Compute SAE activations from hook_resid_pre at each layer
                for layer in range(num_layers):
                    resid_pre = cache[f"blocks.{layer}.hook_resid_pre"]
                    encoded = saes_by_layer[layer]["sae"].encode(resid_pre.to(device)).squeeze(0)
                    if encoded.shape[1] != saes_by_layer[layer]["running_sum"].shape[0]:
                        raise ValueError(
                            f"[Layer {layer}] shape mismatch: got {encoded.shape}, expected {saes_by_layer[layer]['running_sum'].shape[0]}"
                        )
                    saes_by_layer[layer]["running_sum"] += encoded.sum(dim=0)
                    saes_by_layer[layer]["token_count"] += encoded.shape[0]

            del cache
            torch.cuda.empty_cache()

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

    # Save average SAE activations per layer
    model_name = models[i].split("/")[-1]
    for layer in range(num_layers):
        avg_sae = (saes_by_layer[layer]["running_sum"] / saes_by_layer[layer]["token_count"]).detach().cpu().numpy().tolist()
        with open(f"avg_sae_{model_name}_layer{layer}.json", "w") as f:
            json.dump(avg_sae, f)
        print(f"💾 Saved avg SAE activations for layer {layer} to avg_sae_{model_name}_layer{layer}.json")

        del saes_by_layer[layer]["sae"]
        torch.cuda.empty_cache()

    # Final accuracy
    accuracy = sum(r["correct"] for r in results) / len(results)
    print(f"✅ Model Accuracy on ARC-Easy: {accuracy:.2%}")

    with open(f"results_{model_name}.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"💾 Saved prediction results to results_{model_name}.pkl")

    torch.cuda.empty_cache()