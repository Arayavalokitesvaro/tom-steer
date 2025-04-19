import os
import json
import time
import random
import re
import argparse
from tqdm import tqdm
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ----------------------------
# Step 0: Load local GPT-2 model and tokenizer
# ----------------------------

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # Official base model
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# ----------------------------
# Step 1: Parse the ToMi dataset
# ----------------------------

def parse_tomi_dataset(file_path):
    stories = []
    current_story = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Split into line number and content
            split_idx = line.find(' ')
            line_id = int(line[:split_idx])
            content = line[split_idx + 1:]

            # If a new story starts (line_id == 1 and current_story not empty)
            if line_id == 1 and current_story:
                stories.append(_build_story(current_story))
                current_story = []

            current_story.append((line_id, content))

        # Handle last story at EOF
        if current_story:
            stories.append(_build_story(current_story))

    return stories


def _build_story(story_lines):
    context_lines = [s for (_, s) in story_lines[:-1]]
    question_line = story_lines[-1][1]

    try:
        q_part, rest = question_line.split('?', 1)
        question = q_part.strip() + '?'
        answer = rest.strip().split()[0].lower()
    except ValueError:
        question = question_line.strip()
        answer = ""

    return {
        "context": context_lines,
        "question": question,
        "answer": answer
    }

# ----------------------------
# Step 2: Local GPT-2 baseline generation
# ----------------------------

def get_prediction(prompt, features=None, model_id=None, temperature=0.5):
    """
    Generate up to 4 tokens from local GPT-2, approximating Neuronpedia's /api/steer behavior.
    """
    import torch
    import random
    import numpy as np

    # Set random seed for reproducibility (like `seed=16` in API)
    seed = 16
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=4,
        do_sample=True,
        temperature=temperature,
        repetition_penalty=1.1,  # Rough equivalent to freq_penalty=0.3
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode continuation
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_text[len(prompt):].strip()


# ----------------------------
# Step 3: Build prompts and evaluate
# ----------------------------

def build_few_shot_prompt(examples):
    prompt = ""
    for ex in examples:
        context = " ".join(ex["context"])
        prompt += f"{context} {ex['question']} {ex['answer']}\n\n"
    return prompt


def extract_answer_after_last_question(pred_text):
    try:
        q_marks = [m.end() for m in re.finditer(r'\?', pred_text)]
        if not q_marks:
            return pred_text.strip().split()[-1].lower()
        last_pos = q_marks[-1]
        after = pred_text[last_pos:].strip()
        after = re.sub(r'[^\w\s]', '', after)
        after = re.sub(r'\b(?:um|uh|like|you know|so|well)\b', '', after, flags=re.IGNORECASE)
        after = after.strip()
        return after.split()[0].lower() if after else ""
    except:
        return ""  # Return empty string on any parsing failure


def evaluate_with_baseline(data_file, few_shot_n=3, temperature=0.5, num_examples=None):
    data = json.load(open(data_file))
    few_shot = data[:few_shot_n]
    eval_set = data[few_shot_n:]
    
    # Apply num_examples limit if specified
    if num_examples is not None:
        eval_set = eval_set[:num_examples]

    prompt_prefix = build_few_shot_prompt(few_shot)
    correct, total = 0, 0
    results = []  # List to store prediction details

    for item in tqdm(eval_set):
        context = " ".join(item['context'])
        prompt = f"{prompt_prefix}{context} {item['question']}"
        pred = get_prediction(prompt, temperature=temperature)
        pred_token = extract_answer_after_last_question(pred)
        is_correct = (pred_token == item['answer'])
        correct += is_correct
        total += 1
        results.append({
            "context": item['context'],
            "question": item['question'],
            "correct_answer": item['answer'],
            "generated_text": pred,
            "predicted_answer": pred_token,
            "is_correct": is_correct
        })
        time.sleep(1)

    accuracy = correct / total if total else 0
    return accuracy, correct, total, results

# ----------------------------
# Step 4: CLI
# ----------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPT-2 baseline evaluation on ToMi')
    parser.add_argument('--data-file', type=str, required=True,
                        help='Path to JSON file of parsed ToMi examples')
    parser.add_argument('--few-shot', type=int, default=3,
                        help='Number of examples for few-shot prefix')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Sampling temperature')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save output files')
    # Add num-examples argument
    parser.add_argument('--num-examples', type=int, default=None,
                        help='Number of evaluation examples to run (default: all)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Pass num_examples to evaluate function
    acc, corr, tot, results = evaluate_with_baseline(
        args.data_file,
        few_shot_n=args.few_shot,
        temperature=args.temperature,
        num_examples=args.num_examples
    )

    # Update filename to include num_examples if specified
    if args.num_examples:
        shots_suffix = f"{args.few_shot}_examples{args.num_examples}"
    else:
        shots_suffix = f"{args.few_shot}"

    output_txt_path = os.path.join(args.output_dir, f'baseline_{tot}_shots{shots_suffix}.txt')

    with open(output_txt_path, 'w') as out:
        out.write(f"Baseline Accuracy: {acc:.2%} ({corr}/{tot})\n")

    # Save detailed predictions to JSON
    output_json_path = os.path.join(args.output_dir, 'baseline_predictions.json')
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Baseline Accuracy: {acc:.2%} ({corr}/{tot})")
    print(f"Results saved to {output_json_path} and {output_txt_path}.")