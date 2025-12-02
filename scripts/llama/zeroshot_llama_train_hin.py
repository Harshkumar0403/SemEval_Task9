#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import time
import json
import os
import pandas as pd
from sklearn.metrics import f1_score
import traceback

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
API_KEY = "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"      # Replace this
MODEL_NAME = "meta/llama-4-maverick-17b-128e-instruct"

INPUT_FILE = "Dataset/Subset/hin_200.tsv"
PRED_DIR = "llama/train/zeroshot"
PRED_FILE = f"{PRED_DIR}/pred_hin.csv"
EVAL_FILE = f"{PRED_DIR}/eval_hin.txt"

os.makedirs(PRED_DIR, exist_ok=True)

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json"
}

# ---------------------------------------------------------
# SYSTEM PROMPT (Hindi-specific)
# ---------------------------------------------------------
SYSTEM_PROMPT = """
You are an expert social media discourse analyst specializing in Hindi political, religious, and identity-based polarization.
You work as a classifier for the SemEval 2026 Task 9 (Subtask 2): Polarization Type Classification.

Your job is to analyze short Hindi social media posts and identify which explicit types of polarization appear in the text.
Hindi online content often includes sarcasm, hashtags, slogans, caste/religion references, political party names, leader names,
and emojis such as 🚩🙏😂—all of which may signal polarization.

You must detect ONLY polarization directly expressed in the message.

VALID POLARIZATION CATEGORIES:
1. Political
2. Racial/Ethnic
3. Religious
4. Gender/Sexual
5. Other

DEFINITIONS:
• "Political" includes references to political parties, leaders, elections, slogans, government/blame.
• "Religious" includes Hindu–Muslim conflict, caste–religion symbols, gods, rituals, slogans like “जय श्री राम”.
• "Racial/Ethnic" includes caste identity, tribe, regional discrimination, ethnic group hostility.
• "Gender/Sexual" includes misogyny, LGBTQ+ hostility, gender-based insults.
• "Other" includes group-based division not fitting above categories.
• If no polarization exists, output: None.

RULES:
• Multiple labels are allowed.
• Do NOT assume or infer beyond the text.
• Emojis, hashtags, slogans may indicate polarization.
• Return ONLY valid labels: Political, Racial/Ethnic, Religious, Gender/Sexual, Other, None.

OUTPUT FORMAT (STRICT):
{"labels": "Political, Religious"}
{"labels": "None"}
"""

LABEL_LIST = ["Political", "Racial/Ethnic", "Religious", "Gender/Sexual", "Other"]

# ---------------------------------------------------------
# API caller with retry + exponential backoff
# ---------------------------------------------------------
def call_llama(text, max_retries=5):
    user_prompt = f'Text: "{text}"\nReturn labels only in strict JSON format.'

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
            response = requests.post(invoke_url, headers=HEADERS, json=payload)
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                return content
            else:
                print(f"⚠ API error {response.status_code}: {response.text}")
        except Exception as e:
            print("⚠ Exception:", e)
            traceback.print_exc()

        print(f"Retrying in {delay} seconds...")
        time.sleep(delay)
        delay *= 2

    return None

# ---------------------------------------------------------
# Parse JSON output → label list
# ---------------------------------------------------------
def parse_labels(raw):
    if raw is None:
        return ["None"]

    try:
        raw = raw.strip()
        js = raw[raw.find("{"):raw.rfind("}")+1]
        obj = json.loads(js)
        labels = obj["labels"]

        if labels.strip().lower() == "none":
            return ["None"]

        return [x.strip() for x in labels.split(",")]
    except:
        return ["None"]

# ---------------------------------------------------------
# Convert label list to binary vector
# ---------------------------------------------------------
def to_binary_cols(label_list):
    if "None" in label_list:
        return [0, 0, 0, 0, 0]
    return [1 if lbl in label_list else 0 for lbl in LABEL_LIST]

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("\n=== Zero-Shot LLAMA Classification (Hindi) ===")

    df = pd.read_csv(INPUT_FILE, sep="\t")

    preds = []
    gold = []

    for idx, row in df.iterrows():
        text = row["text"]
        gold_labels = row["labels"]

        # Safe gold parsing
        if not isinstance(gold_labels, str) or gold_labels.strip() == "" or gold_labels.lower() == "none":
            gold_list = ["None"]
        else:
            gold_list = [x.strip() for x in gold_labels.split(",")]

        gold_bin = to_binary_cols(gold_list)
        gold.append(gold_bin)

        # LLM call
        out = call_llama(text)
        pred_list = parse_labels(out)
        pred_bin = to_binary_cols(pred_list)
        preds.append(pred_bin)

        time.sleep(0.1)

    # Save predictions
    pred_df = pd.DataFrame(preds, columns=[
        "political", "racial/ethnic", "religious", "gender/sexual", "other"
    ])
    pred_df.insert(0, "id", df["id"])
    pred_df.to_csv(PRED_FILE, index=False)
    print("Saved predictions →", PRED_FILE)

    # Evaluation
    gold_df = pd.DataFrame(gold, columns=[
        "political", "racial/ethnic", "religious", "gender/sexual", "other"
    ])
    pred_vals = pred_df.drop(columns=["id"])

    eval_lines = ["=== CLASSWISE F1 SCORES ==="]
    for col in pred_vals.columns:
        f1 = f1_score(gold_df[col], pred_vals[col], zero_division=0)
        eval_lines.append(f"{col}: {f1:.4f}")

    macro_f1 = f1_score(gold_df, pred_vals, average="macro", zero_division=0)
    eval_lines.append(f"\nMacro F1: {macro_f1:.4f}")

    with open(EVAL_FILE, "w") as f:
        f.write("\n".join(eval_lines))

    print("Saved evaluation →", EVAL_FILE)
    print("\n=== DONE (Hindi) ===\n")


if __name__ == "__main__":
    main()

