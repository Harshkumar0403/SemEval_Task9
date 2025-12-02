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
    raise ValueError("❌ NVIDIA_API_KEY environment variable not set.")

MODEL_NAME = "meta/llama-4-maverick-17b-128e-instruct"

INPUT_FILE = "Dataset/Subset/tur_200.tsv"
PRED_DIR = "llama/train/zeroshot"
PRED_FILE = f"{PRED_DIR}/pred_tur.csv"
EVAL_FILE = f"{PRED_DIR}/eval_tur.txt"

os.makedirs(PRED_DIR, exist_ok=True)

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}

# ---------------------------------------------------------
# TURKISH SYSTEM PROMPT
# ---------------------------------------------------------
SYSTEM_PROMPT = """
You are an expert analyst of Turkish social media discourse for SemEval 2026 Task 9 
(Subtask 2): Polarization Type Classification.

Detect ONLY explicit polarization expressed in the text.

VALID LABELS:
Political, Racial/Ethnic, Religious, Gender/Sexual, Other, None

DEFINITIONS:
• Political: hostility toward political parties, government, politicians, ideology, voters, or institutions.
• Racial/Ethnic: insults or hostility toward ethnic groups, nationalities, immigrants, or minorities.
• Religious: hostility toward religious groups, beliefs, clergy, or practices.
• Gender/Sexual: misogyny, homophobia, gender-based insults, hostility toward LGBTQ+ groups.
• Other: explicit hostility toward any other identifiable social group.
• If no explicit group-directed hostility exists, output: None.

IMPORTANT:
• Turkish posts often use sarcasm, slang, humor, or emojis; label polarization ONLY when group hostility is explicit.
• Mentioning a political party, ethnicity, or religion does NOT automatically imply polarization.
• Multi-label classification is allowed only when justified.
• Emojis do not imply hostility on their own.

OUTPUT FORMAT (strict JSON):
{"labels": "Political, Racial/Ethnic"}
{"labels": "None"}
"""

LABEL_LIST = ["Political", "Racial/Ethnic", "Religious", "Gender/Sexual", "Other"]


# ---------------------------------------------------------
# CALL LLAMA
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
    for _ in range(max_retries):
        try:
            resp = requests.post(invoke_url, headers=HEADERS, json=payload)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            else:
                print(f"⚠ API Error {resp.status_code}: {resp.text}")
        except Exception as e:
            print("⚠ Exception:", e)
            traceback.print_exc()

        print(f"   ↻ Retrying in {delay}s...")
        time.sleep(delay)
        delay *= 2

    return None


# ---------------------------------------------------------
# PARSE MODEL OUTPUT
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
    return [1 if x in lbls else 0 for x in LABEL_LIST]


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("\n=== Zero-Shot LLAMA Classification (Turkish) ===\n")

    df = pd.read_csv(INPUT_FILE, sep="\t")
    preds, gold = [], []

    total = len(df)
    print(f"Processing {total} Turkish samples...\n")

    for idx, row in df.iterrows():
        text = row["text"]
        gold_labels = row["labels"]

        # GOLD
        if not isinstance(gold_labels, str) or gold_labels.strip() == "" or gold_labels.lower() == "none":
            gold_list = ["None"]
        else:
            gold_list = [x.strip() for x in gold_labels.split(",")]

        gold.append(to_binary(gold_list))

        # INFERENCE
        raw = call_llama(text)
        pred_list = parse_labels(raw)
        preds.append(to_binary(pred_list))

        print(f"[{idx+1}/{total}] {row['id']} → {pred_list}")

        time.sleep(0.1)

    # SAVE PREDICTIONS
    pred_df = pd.DataFrame(preds, columns=[
        "political", "racial/ethnic", "religious", "gender/sexual", "other"
    ])
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
    print("\n=== DONE (Turkish) ===\n")


if __name__ == "__main__":
    main()

