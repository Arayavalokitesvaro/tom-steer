import os
import requests
from tqdm import tqdm

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
                # Process the previous story
                context_lines = [s for (i, s) in current_story[:-1]]
                question_line = current_story[-1][1]

                try:
                    q_part, rest = question_line.split('?', 1)
                    question = q_part.strip() + '?'
                    answer = rest.strip().split()[0]
                except:
                    question = question_line.strip()
                    answer = ""

                stories.append({
                    "context": context_lines,
                    "question": question,
                    "answer": answer
                })

                # Start new story
                current_story = []

            current_story.append((line_id, content))

        # Handle last story at EOF
        if current_story:
            context_lines = [s for (i, s) in current_story[:-1]]
            question_line = current_story[-1][1]

            try:
                q_part, rest = question_line.split('?', 1)
                question = q_part.strip() + '?'
                answer = rest.strip().split()[0]
            except:
                question = question_line.strip()
                answer = ""

            stories.append({
                "context": context_lines,
                "question": question,
                "answer": answer
            })

    return stories

# ----------------------------
# Step 2: Call Neuronpedia /api/steer
# ----------------------------

def get_prediction(prompt, features, model_id="gpt2-small", temperature=0.5):
    url = "https://www.neuronpedia.org/api/steer"
    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "prompt": prompt,
        "modelId": model_id,
        "features": features,
        "temperature": temperature,
        "n_tokens": 4,
        "freq_penalty": 0.3,
        "seed": 16,
        "strength_multiplier": 1.0,
    }
    print("Payload: ", payload)
    while True:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            print("Result: ", result)
            return result["STEERED"].strip()
        elif response.status_code == 429:  # Too Many Requests
            print("Rate limit exceeded. Retrying after a delay...")
            time.sleep(12)  # Wait for 10 seconds before retrying
        elif response.status_code == 500: 
            print("Unknown Error. Retrying after a delay...")
            time.sleep(12)  # Wait for 10 seconds before retrying
        else:
            print("API error:", response.status_code, response.text)
            return ""

# ----------------------------
# Step 3: Evaluate predictions
# ----------------------------

def build_few_shot_prompt(examples):
    few_shot_prompt = ""
    for example in examples:
        context = " ".join(example["context"])
        question = example["question"]
        answer = example["answer"]
        few_shot_prompt += f"{context} {question} {answer}\n\n"
    return few_shot_prompt

import re

import re
import time
import random
import json
import argparse

def extract_answer_after_last_question(pred_text):
    """
    Extract the first word that comes immediately after the **last** question mark in the model's output.
    """
    # Find all positions of question marks
    question_positions = [m.end() for m in re.finditer(r'\?', pred_text)]

    if not question_positions:
        # Fallback: return last word
        return pred_text.strip().split()[-1].lower()

    # Extract substring starting after the last '?'
    last_qmark_pos = question_positions[-1]
    after_q = pred_text[last_qmark_pos:].strip()
    # Remove punctuation and token fillers
    after_q = re.sub(r'[^\w\s]', '', after_q)  # Remove punctuation
    after_q = re.sub(r'\b(?:um|uh|like|you know|so|well)\b', '', after_q, flags=re.IGNORECASE)  # Remove fillers
    after_q = after_q.strip()
    # Return first word after last '?'
    return after_q.split()[0].strip().lower() if after_q else ""



def evaluate_with_steering(data, features, modelid="gpt2-small"):
    few_shot_prompt = build_few_shot_prompt(data[:3])  # first 3 examples
    eval_data = data[3:]  # remaining examples

    correct = 0
    total = 0

    for item in tqdm(eval_data):
        context = " ".join(item["context"])
        question = item["question"]
        prompt = f"{few_shot_prompt}{context} {question}"

        gold = item["answer"].lower()
        pred = get_prediction(prompt, features, model_id=modelid)
        print(f"Response: {pred}")

        pred_token = extract_answer_after_last_question(pred)

        if pred_token == gold:
            correct += 1
        else:
            print(f"❌ {question}\n   True: {gold} | Pred: {pred_token}\n   → Full: {pred}")

        total += 1
        time.sleep(12)

    accuracy = correct / total
    print(f"\n✅ Steering Accuracy: {accuracy:.2%} ({correct}/{total})")
    return accuracy, correct, total


# ----------------------------
# Step 4: Main runner
# ----------------------------

if __name__ == "__main__":
    # val_data = parse_tomi_dataset("../feature-mining/ToMi/data/val.txt")
    # Parse command-line arguments
    # parser = argparse.ArgumentParser(description="Steering evaluation script.")
    # parser.add_argument("--modelid", type=str, required=True, help="Model ID (e.g., gpt2-small)")
    # parser.add_argument("--layer", type=str, required=True, help="Layer to steer (e.g., 11-res-jb)")
    # parser.add_argument("--index", type=int, required=True, help="Feature index to steer")
    # parser.add_argument("--strength", type=float, required=True, help="Steering strength")
    # args = parser.parse_args()
    def run_evaluation(model, layer, index, strength):
        output_filename = f"./output/accuracy_{model}_{layer}_index{feature}_strength{strength}.txt"
        if os.path.exists(output_filename):
            print(f"Output file {output_filename} already exists. Skipping...")
            return
        # Check if the output directory exists, if not create it
        print(f"Running evaluation for model: {model}, layer: {layer}, index: {index}, strength: {strength}")
        # Parse the TOMI dataset
        # val_data = parse_tomi_dataset("../feature-mining/ToMi/data/val.txt")
        # return val_data
        # Use the parsed arguments
        features = [
            {
                "modelId": model,
                "layer": layer,
                "index": feature,
                "strength": strength
            }
        ]
        # # Randomly select 100 entries from val_data
        # small_val_data = random.sample(val_data, 100)

        # # Dump the selected entries into a JSON file
        # with open("small_val_data.json", "w") as f:
        #     json.dump(small_val_data, f, indent=4)
        # exit(0)
        # Load the small validation data from the JSON file
        with open("small_val_data.json", "r") as f:
            val_data = json.load(f)
        # Example: steer toward feature 3124 in layer 20 of gemma-2b SAE
        # features = [
        #     {
        #         "modelId": "gpt2-small",
        #         "layer": "11-res-jb",
        #         "index": 89,
        #         "strength": 2
        #     }
        # ]

        accuracy, correct, total = evaluate_with_steering(val_data, features, modelid=model)
        # Save the accuracy result in a file
        
        with open(output_filename, "w") as f:
            f.write(f"Steering Accuracy: {accuracy:.2%} ({correct}/{total})\n")
        print(f"Accuracy result saved to {output_filename}")
    import pandas as pd
    df = pd.read_csv("./steer-list.csv")
    for _, row in df.iterrows():
        model = row["model"]
        layer = row["layer"]
        feature = row["feature"]
        for strength in [5.0, 10.0, 15.0]:
            run_evaluation(model, layer, feature, strength)
