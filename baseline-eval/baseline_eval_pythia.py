import os
import json
import time
import random
import re
import argparse
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------
# Step 0: Parse command-line arguments
# ----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Pythia baseline evaluation on ToMi')
    parser.add_argument('--data-file', type=str, required=True,
                        help='Path to JSON file of parsed ToMi examples')
    parser.add_argument('--few-shot', type=int, default=3,
                        help='Number of examples for few-shot prefix')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Sampling temperature')
    parser.add_argument('--model-id', type=str, default='EleutherAI/pythia-1b',
                        help='Huggingface model ID for Pythia')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory to save results')
    return parser.parse_args()

# ----------------------------
# Step 1: Load tokenizer and model
# ----------------------------

def load_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()
    return tokenizer, model

# ----------------------------
# Step 2: Parse the ToMi dataset
# ----------------------------

def parse_tomi_dataset(file_path):
    stories = []
    current_story = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            split_idx = line.find(' ')
            line_id = int(line[:split_idx])
            content = line[split_idx + 1:]

            if line_id == 1 and current_story:
                stories.append(_build_story(current_story))
                current_story = []

            current_story.append((line_id, content))

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
# Step 3: Generation and Evaluation
# ----------------------------

def get_prediction(prompt, tokenizer, model, temperature=0.5):
    seed = 16
    torch.manual_seed(seed)
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=4,
        do_sample=True,
        temperature=temperature,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_text[len(prompt):].strip()


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
        return ""


def build_few_shot_prompt(examples):
    prompt = ""
    for ex in examples:
        context = " ".join(ex["context"])
        prompt += f"{context} {ex['question']} {ex['answer']}\n\n"
    return prompt


def evaluate_with_baseline(data_file, tokenizer, model, few_shot_n=3, temperature=0.5):
    data = json.load(open(data_file))
    few_shot = data[:few_shot_n]
    eval_set = data[few_shot_n:]

    prompt_prefix = build_few_shot_prompt(few_shot)
    correct, total = 0, 0

    for item in tqdm(eval_set):
        context = " ".join(item['context'])
        prompt = f"{prompt_prefix}{context} {item['question']}"
        pred = get_prediction(prompt, tokenizer, model, temperature=temperature)
        pred_token = extract_answer_after_last_question(pred)
        if pred_token == item['answer']:
            correct += 1
        total += 1
        time.sleep(1)

    accuracy = correct / total if total else 0
    return accuracy, correct, total

# ----------------------------
# Step 4: Main entry point
# ----------------------------

if __name__ == '__main__':
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer, model = load_model_and_tokenizer(args.model_id)

    acc, corr, tot = evaluate_with_baseline(
        args.data_file,
        tokenizer,
        model,
        few_shot_n=args.few_shot,
        temperature=args.temperature
    )

    result_str = f"Baseline Accuracy: {acc:.2%} ({corr}/{tot})"
    print(result_str)

    out_path = os.path.join(args.output_dir, f"baseline_{tot}_shots{args.few_shot}.txt")
    with open(out_path, 'w') as out_f:
        out_f.write(result_str + "\n")
    print(f'Results saved to {out_path}.')
