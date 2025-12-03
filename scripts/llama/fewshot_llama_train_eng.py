#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, time, random, traceback
import pandas as pd
import requests
from sklearn.metrics import f1_score

# =========================================================
# CONFIG
# =========================================================
API_KEY = os.getenv("NVIDIA_API_KEY")
if not API_KEY:
    raise ValueError("❌ NVIDIA_API_KEY environment variable not set.")

MODEL_NAME = "meta/llama-4-maverick-17b-128e-instruct"

TRAIN_FILE = "Dataset/Processed/eng.tsv"
SUBSET_FILE = "Dataset/Subset/eng_200.tsv"

OUTPUT_DIR = "llama/fewshot"
PRED_FILE = f"{OUTPUT_DIR}/pred_eng.csv"
EVAL_FILE = f"{OUTPUT_DIR}/eval_eng.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}

# =========================================================
# LABEL ORDER
# =========================================================
LABEL_LIST = ["Political", "Racial/Ethnic", "Religious", "Gender/Sexual", "Other"]


# =========================================================
# HELPER: Convert labels → binary vector
# =========================================================
def to_binary(lbls):
    if "None" in lbls:
        return [0, 0, 0, 0, 0]
    return [1 if x in lbls else 0 for x in LABEL_LIST]


# =========================================================
# STEP 1: Load FULL train set for FEW-SHOT demonstration selection
# =========================================================
df_full = pd.read_csv(TRAIN_FILE, sep="\t")

# Create buckets
buckets = {
    "Political": [],
    "Racial/Ethnic": [],
    "Religious": [],
    "Gender/Sexual": [],
    "Other": [],
    "None": []
}

for idx, row in df_full.iterrows():
    labels = row["labels"]
    if not isinstance(labels, str) or labels.strip() == "" or labels.lower() == "none":
        buckets["None"].append(row)
        continue
    
    parts = [x.strip() for x in labels.split(",")]
    
    # Put this sample into ALL label buckets because it's multi-label
    for p in parts:
        buckets[p].append(row)

# Now we sample according to required distribution
sample_plan = {
    "Political": 2,
    "Racial/Ethnic": 2,
    "Religious": 2,
    "Gender/Sexual": 1,
    "Other": 1,
    "None": 2
}

fewshot_examples = []

for label, count in sample_plan.items():
    if len(buckets[label]) >= count:
        fewshot_examples.extend(random.sample(buckets[label], count))
    else:
        # backup sampling if bucket has < count
        fewshot_examples.extend(buckets[label])

# =========================================================
# BUILD FEW-SHOT PROMPT
# =========================================================
fewshot_text = ""

for i, row in enumerate(fewshot_examples, start=1):
    lbl = row["labels"]
    if not isinstance(lbl, str) or lbl.strip() == "":
        lbl = "None"

    fewshot_text += (
        f"Example {i}:\n"
        f"Text: \"{row['text']}\"\n"
        f"Labels: {lbl}\n\n"
    )


# SYSTEM PROMPT FOR ENGLISH
SYSTEM_PROMPT = f"""
You are an expert analyst of English social media content working on SemEval 2026 Task 9
Subtask 2: Polarization Type Classification.

VALID LABELS:
Political, Racial/Ethnic, Religious, Gender/Sexual, Other, None

You must detect ONLY explicit group-directed hostility, not implied meaning.

Few-shot examples are provided below:

{fewshot_text}

Now classify the following text. Return ONLY a JSON object:
{{"labels": "<comma-separated list or None>"}}
"""


# =========================================================
# CALL LLAMA API
# =========================================================
def call_llama(text, max_retries=5):
    user_prompt = (
        f'Text: "{text}"\n'
        f'Return ONLY the JSON object in this format:\n'
        f'{{"labels": "..."}}'
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 128,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False
    }

    delay = 1.0

    for _ in range(max_retries):
        try:
            resp = requests.post(invoke_url, headers=HEADERS, json=payload)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            else:
                print(f"⚠ API ERROR {resp.status_code}: {resp.text}")
        except Exception as e:
            print("⚠ Exception:", e)
            traceback.print_exc()

        print(f"↻ Retrying in {delay}s...")
        time.sleep(delay)
        delay *= 2

    return None


# =========================================================
# PARSE LLaMA JSON OUTPUT
# =========================================================
def parse_labels(raw):
    if raw is None:
        return ["None"]
    try:
        js = raw[raw.find("{"): raw.rfind("}") + 1]
        obj = json.loads(js)
        labels = obj["labels"]

        if labels.lower() == "none":
            return ["None"]

        return [x.strip() for x in labels.split(",")]
    except:
        return ["None"]


# =========================================================
# MAIN FEW-SHOT INFERENCE
# =========================================================
def main():

    print("\n=== FEW-SHOT LLAMA Classification (English) ===")

    df = pd.read_csv(SUBSET_FILE, sep="\t")

    preds = []
    gold = []

    total = len(df)
    print(f"Processing {total} English samples...\n")

    for idx, row in df.iterrows():

        text = row["text"]
        gold_lbl = row["labels"]

        # GOLD PARSE
        if not isinstance(gold_lbl, str) or gold_lbl.strip() == "" or gold_lbl.lower() == "none":
            gold_list = ["None"]
        else:
            gold_list = [x.strip() for x in gold_lbl.split(",")]

        gold.append(to_binary(gold_list))

        raw = call_llama(text)
        pred_list = parse_labels(raw)
        preds.append(to_binary(pred_list))

        print(f"[{idx+1}/{total}] {row['id']} → {pred_list}")
        time.sleep(0.1)

    # SAVE PRED CSV
    pred_df = pd.DataFrame(
        preds,
        columns=["political", "racial/ethnic", "religious", "gender/sexual", "other"]
    )
    pred_df.insert(0, "id", df["id"])
    pred_df.to_csv(PRED_FILE, index=False)

    print("\nSaved predictions →", PRED_FILE)

    # EVALUATION
    gold_df = pd.DataFrame(gold, columns=pred_df.columns[1:])
    pred_vals = pred_df.drop(columns=["id"])

    lines = ["=== CLASSWISE F1 SCORES ==="]

    for col in pred_vals.columns:
        f1 = f1_score(gold_df[col], pred_vals[col], zero_division=0)
        lines.append(f"{col}: {f1:.4f}")

    macro_f1 = f1_score(gold_df, pred_vals, average="macro", zero_division=0)
    lines.append(f"\nMacro F1: {macro_f1:.4f}")

    with open(EVAL_FILE, "w") as f:
        f.write("\n".join(lines))

    print("Saved evaluation →", EVAL_FILE)
    print("\n=== DONE (FEW-SHOT English) ===\n")


if __name__ == "__main__":
    main()

