#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import time
import json
import os
import pandas as pd
from sklearn.metrics import f1_score, classification_report
import traceback
import random

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
API_KEY = "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"  
MODEL_NAME = "meta/llama-4-maverick-17b-128e-instruct"

INPUT_FILE = "Dataset/Subset/eng_200.tsv"
PRED_DIR = "llama/train/zeroshot"
PRED_FILE = f"{PRED_DIR}/pred_eng.csv"
EVAL_FILE = f"{PRED_DIR}/eval_eng.txt"

os.makedirs(PRED_DIR, exist_ok=True)

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json"
}

# ---------------------------------------------------------
# SYSTEM PROMPT for English Zero-Shot Classification
# ---------------------------------------------------------
SYSTEM_PROMPT = """
You are an expert classifier for the SemEval 2026 Task 9 on polarization type detection.

Your job is to analyze a short English social media message and identify which, if any, of the following polarization types are explicitly expressed in the text:

1. Political
2. Racial/Ethnic
3. Religious
4. Gender/Sexual
5. Other

RULES:
- Predict ONLY the types of polarization expressed directly in the text.
- If the message contains no polarization, output: None
- Do NOT infer or imagine polarization that is not clearly present.
- Ignore the opinion of the author, the reader, or people mentioned indirectly.
- Your output MUST be a comma-separated list using ONLY these labels:
  Political, Racial/Ethnic, Religious, Gender/Sexual, Other, None

FORMAT:
Return ONLY this JSON structure:
{"labels": "Political, Religious"}
or:
{"labels": "None"}
"""

LABEL_LIST = ["Political", "Racial/Ethnic", "Religious", "Gender/Sexual", "Other"]

# ---------------------------------------------------------
# FUNCTION: Call LLaMA with retry + backoff
# ---------------------------------------------------------
def call_llama(text, max_retries=5):
    user_prompt = f'Text: "{text}"\nReturn labels only in required JSON format.'

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

    delay = 1.0  # base backoff
    for attempt in range(max_retries):
        try:
            response = requests.post(invoke_url, headers=HEADERS, json=payload)
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return content

            else:
                print(f"⚠ API error {response.status_code}: {response.text}")
        except Exception as e:
            print("⚠ Exception:", e)
            traceback.print_exc()

        # Retry with exponential backoff
        print(f"Retrying in {delay} seconds...")
        time.sleep(delay)
        delay *= 2

    print("❌ Failed after max retries.")
    return None


# ---------------------------------------------------------
# FUNCTION: Parse model output → list of labels
# ---------------------------------------------------------
def parse_labels(raw_output):
    if raw_output is None:
        return ["None"]

    try:
        # extract JSON
        raw_output = raw_output.strip()
        json_start = raw_output.find("{")
        json_end = raw_output.rfind("}") + 1
        json_str = raw_output[json_start:json_end]

        obj = json.loads(json_str)
        labels = obj["labels"]

        if labels.strip().lower() == "none":
            return ["None"]

        return [l.strip() for l in labels.split(",")]

    except Exception as e:
        print("⚠ Parsing error:", e)
        print("RAW OUTPUT:", raw_output)
        return ["None"]


# ---------------------------------------------------------
# FUNCTION: Convert label list → 5 binary columns
# ---------------------------------------------------------
def to_binary_cols(label_list):
    if "None" in label_list:
        return [0, 0, 0, 0, 0]

    out = []
    for lbl in LABEL_LIST:
        out.append(1 if lbl in label_list else 0)
    return out


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("\n=== Zero-Shot Classification with LLAMA (English) ===")

    df = pd.read_csv(INPUT_FILE, sep="\t")

    preds = []
    gold = []

    for idx, row in df.iterrows():
        text = row["text"]
        gold_labels = row["labels"]

        # ---- FIX: Make gold label parsing safe ----
        if not isinstance(gold_labels, str) or gold_labels.strip() == "" or gold_labels.strip().lower() == "none":
            gold_list = ["None"]
        else:
            gold_list = [x.strip() for x in gold_labels.split(",")]

        gold_binary = to_binary_cols(gold_list)
        gold.append(gold_binary)

        # Call LLaMA
        raw_out = call_llama(text)
        label_list = parse_labels(raw_out)
        pred_binary = to_binary_cols(label_list)
        preds.append(pred_binary)

        print(f"[{idx+1}/{len(df)}] ID={row['id']}")
        print("TEXT:", text)
        print("PRED:", label_list, pred_binary)
        print("GOLD:", gold_labels, gold_binary)
        print()

        time.sleep(0.1)


    # Convert to DataFrame
    pred_df = pd.DataFrame(preds, columns=[
        "political", "racial/ethnic", "religious", "gender/sexual", "other"
    ])
    pred_df.insert(0, "id", df["id"])

    pred_df.to_csv(PRED_FILE, index=False)
    print(f"Predictions saved → {PRED_FILE}")

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------
    gold_arr = pd.DataFrame(gold, columns=[
        "political", "racial/ethnic", "religious", "gender/sexual", "other"
    ])
    pred_arr = pred_df.drop(columns=["id"])

    eval_text = []

    eval_text.append("=== CLASSWISE F1 SCORES ===")
    for i, col in enumerate(pred_arr.columns):
        f1 = f1_score(gold_arr[col], pred_arr[col], zero_division=0)
        eval_text.append(f"{col}: {f1:.4f}")

    macro_f1 = f1_score(gold_arr, pred_arr, average="macro", zero_division=0)
    eval_text.append(f"\nMacro F1: {macro_f1:.4f}\n")

    # Save evaluation file
    with open(EVAL_FILE, "w") as f:
        f.write("\n".join(eval_text))

    print("\nEvaluation saved →", EVAL_FILE)
    print("\n=== DONE ===\n")


if __name__ == "__main__":
    main()

