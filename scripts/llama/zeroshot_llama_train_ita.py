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

INPUT_FILE = "Dataset/Subset/ita_200.tsv"
PRED_DIR = "llama/train/zeroshot"
PRED_FILE = f"{PRED_DIR}/pred_ita.csv"
EVAL_FILE = f"{PRED_DIR}/eval_ita.txt"

os.makedirs(PRED_DIR, exist_ok=True)

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json"
}

# ---------------------------------------------------------
# Italian SYSTEM PROMPT
# ---------------------------------------------------------
SYSTEM_PROMPT = """
You are an expert analyst of Italian social media discourse, specializing in detecting political,
racial/ethnic, religious, gender-based, and other forms of polarization. You work as a classifier
for the SemEval 2026 Task 9 (Subtask 2): Polarization Type Classification.

Your task is to read short Italian social media posts and identify explicit polarization.

VALID LABELS:
Political, Racial/Ethnic, Religious, Gender/Sexual, Other, None

DEFINITIONS:
• Political: hostility toward political parties, leaders, government, ideology.
• Racial/Ethnic: hostility toward immigrants, minorities, nationalities, or regions.
• Religious: insults or division involving religion, church, believers.
• Gender/Sexual: misogyny, homophobia, transphobia, gender-based insults.
• Other: group hostility not covered above.
• If no explicit hostility exists, output: None.

RULES:
• Multiple labels may apply.
• Irony, slang, and emojis do not imply polarization unless hostility is expressed.
• Mentions of immigration, gender, or religion WITHOUT hostility do NOT count.
• Output MUST be JSON:
  {"labels": "Racial/Ethnic, Religious"}
  {"labels": "None"}
"""

LABEL_LIST = ["Political", "Racial/Ethnic", "Religious", "Gender/Sexual", "Other"]

# ---------------------------------------------------------
# API CALL
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

        print(f"   ↻ Retrying in {delay} sec...")
        time.sleep(delay)
        delay *= 2

    return None

# ---------------------------------------------------------
# PARSE LABEL JSON
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# BINARY ENCODING
# ---------------------------------------------------------
def to_binary(lbls):
    if "None" in lbls:
        return [0, 0, 0, 0, 0]
    return [1 if x in lbls else 0 for x in ["Political", "Racial/Ethnic", "Religious", "Gender/Sexual", "Other"]]

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("\n=== Zero-Shot LLAMA Classification (Italian) ===\n")

    df = pd.read_csv(INPUT_FILE, sep="\t")
    preds, gold = [], []
    total = len(df)

    print(f"Processing {total} Italian samples...\n")

    for idx, row in df.iterrows():
        text = row["text"]
        gold_labels = row["labels"]

        # gold parsing
        if not isinstance(gold_labels, str) or gold_labels.lower() == "none" or gold_labels.strip() == "":
            gold_list = ["None"]
        else:
            gold_list = [x.strip() for x in gold_labels.split(",")]

        gold.append(to_binary(gold_list))

        # model inference
        raw = call_llama(text)
        pred_list = parse_labels(raw)
        preds.append(to_binary(pred_list))

        print(f"[{idx+1}/{total}] {row['id']} → {pred_list}")

        time.sleep(0.1)

    pred_df = pd.DataFrame(preds, columns=[
        "political", "racial/ethnic", "religious", "gender/sexual", "other"
    ])
    pred_df.insert(0, "id", df["id"])
    pred_df.to_csv(PRED_FILE, index=False)

    print("\nSaved predictions →", PRED_FILE)

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
    print("\n=== DONE (Italian) ===\n")


if __name__ == "__main__":
    main()

