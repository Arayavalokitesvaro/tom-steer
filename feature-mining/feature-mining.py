from datasets import load_dataset
import pandas as pd
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from sae_lens.sae import SAE

device = "cuda" if torch.cuda.is_available() else "cpu"

# ARC-Easy dataset
ds = load_dataset("allenai/ai2_arc", "ARC-Easy")["train"]
df = pd.DataFrame(ds)

# Model and SAE configs
models = ['gpt2', 'google/gemma-2-2b', 'EleutherAI/pythia-70m-deduped']
saes = ['gpt2-small-res-jb', 'gemma-scope-2b-pt-res', 'pythia-70m-deduped-res-sm']

for i in range(len(models)):
    print(f"\n==== Evaluating Model: {models[i]} ====")
    
    # Load model
    model = HookedTransformer.from_pretrained(models[i], device=device)
    tokenizer = model.tokenizer
    num_layers = model.cfg.n_layers

    # Load SAE per layer
    this_sae = []
    for layer in range(num_layers):
        sae, _, sparsity = SAE.from_pretrained(
            release=saes[i],
            sae_id=f"blocks.{layer}.hook_resid_pre",
            device=device,
        )
        this_sae.append(sae)
        print(f"Layer {layer}: SAE loaded with sparsity {sparsity.mean().item():.2f}")

    results = []
    model.eval()

    for _, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]
        choices = row["choices"]["text"]
        labels = row["choices"]["label"]
        answer_key = row["answerKey"]
        
        logprobs = []
        sae_activations = []

        for choice in choices:
            prompt = f"{question} {choice}"
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            input_len = input_ids.shape[1]
            
            # Run model + capture cache
            with torch.no_grad():
                _, cache = model.run_with_cache(prompt)

            # Logprob of answer tokens
            logits = model(prompt).logits[:, :-1, :]  # Exclude final token
            probs = torch.nn.functional.log_softmax(logits, dim=-1)
            target_ids = input_ids[:, 1:]  # shift target
            choice_logprob = torch.gather(probs, 2, target_ids.unsqueeze(-1)).sum().item()
            logprobs.append(choice_logprob)

            # SAE features from last token of each layer
            sae_feats = []
            for l in range(num_layers):
                resid_pre = cache[f"blocks.{l}.hook_resid_pre"][-1].unsqueeze(0)  # last token
                feats = this_sae[l].encode(resid_pre.to(device)).squeeze(0).cpu().numpy()
                sae_feats.append(feats)
            sae_activations.append(sae_feats)

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
            "logprobs": logprobs,
            "sae_activations": sae_activations  # [num_choices][num_layers][features]
        })

    # Report accuracy
    accuracy = sum(r["correct"] for r in results) / len(results)
    print(f"✅ Model Accuracy on ARC-Easy: {accuracy:.2%}")

    # Optional: Save results
    with open(f"results_{models[i].split('/')[-1]}.pkl", "wb") as f:
       import pickle; pickle.dump(results, f)
