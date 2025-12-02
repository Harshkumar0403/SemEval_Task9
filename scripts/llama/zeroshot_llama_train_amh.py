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

INPUT_FILE = "Dataset/Subset/amh_200.tsv"
PRED_DIR = "llama/train/zeroshot"
PRED_FILE = f"{PRED_DIR}/pred_amh.csv"
EVAL_FILE = f"{PRED_DIR}/eval_amh.txt"

os.makedirs(PRED_DIR, exist_ok=True)

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json"
}

# ---------------------------------------------------------
# Amharic system prompt
# ---------------------------------------------------------
SYSTEM_PROMPT = """
You are an expert analyst of Amharic social media discourse, specializing in detecting political,
ethnic, and group-based polarization. You work as a classifier for the SemEval 2026 Task 9
(Subtask 2): Polarization Type Classification.

Your goal is to analyze short Amharic social media posts and detect explicit types of polarization.

Valid labels:
Political, Racial/Ethnic, Religious, Gender/Sexual, Other, None

Guidelines:
• "Political": government, political parties, protests, conflict, political blame.
• "Racial/Ethnic": ethnicity-based hostility, regional tension, ethnic slurs.
• "Religious": only if division about religion is expressed.
• "Gender/Sexual": only if hostility targets gender or sexuality.
• "Other": group-based division not fitting the above.
• If no polarization exists, output: None.

Rules:
• Multiple labels may apply.
• Do NOT infer hidden intentions.
• Interpret emojis, hashtags, slogans correctly.
• Output MUST be JSON exactly like:
  {"labels": "Political, Racial/Ethnic"}
  {"labels": "None"}
"""

LABEL_LIST = ["Political", "Racial/Ethnic", "Religious", "Gender/Sexual", "Other"]

# ---------------------------------------------------------
# API CALLER with retry + backoff + progress prints
# ---------------------------------------------------------
def call_llama(text, max_retries=5):
    user_prompt = f'Text: "{text}"\nReturn only the JSON object with labels.'

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
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

        print(f"   ↻ Retry in {delay} sec")
        time.sleep(delay)
        delay *= 2

    return None

# ---------------------------------------------------------
# PARSER
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
# LABEL → BINARY
# ---------------------------------------------------------
def to_binary_cols(label_list):
    if "None" in label_list:
        return [0, 0, 0, 0, 0]
    return [1 if lbl in label_list else 0 for lbl in LABEL_LIST]

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("\n=== Zero-Shot LLAMA Classification (Amharic) ===\n")
    df = pd.read_csv(INPUT_FILE, sep="\t")

    preds, gold = [], []

    total = len(df)
    print(f"Processing {total} samples...\n")

    for idx, row in df.iterrows():
        text = row["text"]
        gold_labels = row["labels"]

        # SAFETY check for gold labels
        if not isinstance(gold_labels, str) or gold_labels.strip() == "" or gold_labels.lower() == "none":
            gold_list = ["None"]
        else:
            gold_list = [x.strip() for x in gold_labels.split(",")]

        gold_bin = to_binary_cols(gold_list)
        gold.append(gold_bin)

        # Call model
        out = call_llama(text)
        pred_list = parse_labels(out)
        pred_bin = to_binary_cols(pred_list)
        preds.append(pred_bin)

        # Terminal progress:
        print(f"[{idx+1}/{total}] {row['id']}  →  {pred_list}")

        time.sleep(0.1)

    # SAVE PREDICTIONS
    pred_df = pd.DataFrame(preds, columns=[
        "political", "racial/ethnic", "religious", "gender/sexual", "other"
    ])
    pred_df.insert(0, "id", df["id"])
    pred_df.to_csv(PRED_FILE, index=False)
    print("\nSaved predictions →", PRED_FILE)

    # EVALUATION
    gold_df = pd.DataFrame(gold, columns=[
        "political", "racial/ethnic", "religious", "gender/sexual", "other"
    ])
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
    print("\n=== DONE (Amharic) ===\n")


if __name__ == "__main__":
    main()

