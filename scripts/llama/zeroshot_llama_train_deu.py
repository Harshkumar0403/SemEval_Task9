#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests, time, json, os, traceback
import pandas as pd
from sklearn.metrics import f1_score

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
API_KEY = os.getenv("NVIDIA_API_KEY")
if not API_KEY:
    raise ValueError("❌ ERROR: NVIDIA_API_KEY environment variable is not set.")

MODEL_NAME = "meta/llama-4-maverick-17b-128e-instruct"

INPUT_FILE = "Dataset/Subset/deu_200.tsv"
PRED_DIR = "llama/train/zeroshot"
PRED_FILE = f"{PRED_DIR}/pred_deu.csv"
EVAL_FILE = f"{PRED_DIR}/eval_deu.txt"

os.makedirs(PRED_DIR, exist_ok=True)

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json"
}

# ---------------------------------------------------------
# German SYSTEM PROMPT
# ---------------------------------------------------------
SYSTEM_PROMPT = """
You are an expert analyst of German social media discourse, specializing in detecting political,
racial/ethnic, religious, gender-based, and other forms of polarization. You work as a classifier
for the SemEval 2026 Task 9 (Subtask 2): Polarization Type Classification.

Your task is to read short German social media posts and identify explicit polarization.

VALID LABELS:
Political, Racial/Ethnic, Religious, Gender/Sexual, Other, None

DEFINITIONS:
• Political: parties, politicians, government, ideological hostility, elections, protests.
• Racial/Ethnic: hostility based on nationality, ethnicity, migration background, skin color.
• Religious: insults or division targeting religious groups.
• Gender/Sexual: misogyny, gendered insults, LGBTQ+ hostility.
• Other: group-based hostility not fitting above categories.
• If no polarization exists, output: None.

RULES:
• Multiple labels may apply.
• If a topic is mentioned without hostility or division, output: None.
• Understand sarcasm, compound nouns, hashtags, and emojis.
• Output MUST be JSON:
  {"labels": "Political, Religious"}
  {"labels": "None"}
"""

LABEL_LIST = ["Political", "Racial/Ethnic", "Religious", "Gender/Sexual", "Other"]

# ---------------------------------------------------------
# API CALLER (retry + progress)
# ---------------------------------------------------------
def call_llama(text, max_retries=5):
    user_prompt = f'Text: "{text}"\nReturn ONLY the JSON object with labels.'

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
    for attempt in range(max_retries):
        try:
            r = requests.post(invoke_url, headers=HEADERS, json=payload)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            else:
                print(f"⚠ API Error {r.status_code}: {r.text}")
        except Exception as e:
            print("⚠ Exception:", e)
            traceback.print_exc()

        print(f"   ↻ Retrying in {delay} seconds...")
        time.sleep(delay)
        delay *= 2

    return None

# ---------------------------------------------------------
# PARSE LLaMA OUTPUT
# ---------------------------------------------------------
def parse_labels(raw):
    if raw is None:
        return ["None"]
    try:
        raw = raw.strip()
        js = raw[raw.find("{"):raw.rfind("}")+1]
        obj = json.loads(js)
        labels = obj["labels"]

        if labels.lower() == "none":
            return ["None"]

        return [x.strip() for x in labels.split(",")]
    except:
        return ["None"]

# ---------------------------------------------------------
# CONVERT LABELS TO BINARY
# ---------------------------------------------------------
def to_binary_cols(label_list):
    if "None" in label_list:
        return [0, 0, 0, 0, 0]
    return [1 if lbl in label_list else 0 for lbl in LABEL_LIST]


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("\n=== Zero-Shot LLAMA Classification (German) ===\n")

    df = pd.read_csv(INPUT_FILE, sep="\t")

    preds, gold = [], []
    total = len(df)

    print(f"Processing {total} German samples...\n")

    for idx, row in df.iterrows():
        text = row["text"]
        gold_labels = row["labels"]

        # safe gold parsing
        if not isinstance(gold_labels, str) or gold_labels.lower() == "none" or gold_labels.strip() == "":
            gold_list = ["None"]
        else:
            gold_list = [x.strip() for x in gold_labels.split(",")]

        gold_bin = to_binary_cols(gold_list)
        gold.append(gold_bin)

        # call LLaMA
        raw = call_llama(text)
        pred_list = parse_labels(raw)
        pred_bin = to_binary_cols(pred_list)
        preds.append(pred_bin)

        # terminal progress
        print(f"[{idx+1}/{total}] {row['id']} → {pred_list}")

        time.sleep(0.1)

    # save predictions
    pred_df = pd.DataFrame(preds, columns=[
        "political", "racial/ethnic", "religious", "gender/sexual", "other"
    ])
    pred_df.insert(0, "id", df["id"])
    pred_df.to_csv(PRED_FILE, index=False)
    print("\nSaved predictions →", PRED_FILE)

    # evaluation
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
    print("\n=== DONE (German) ===\n")


if __name__ == "__main__":
    main()

