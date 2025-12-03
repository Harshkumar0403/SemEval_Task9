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

TRAIN_FILE = "Dataset/Processed/urd.tsv"
SUBSET_FILE = "Dataset/Subset/urd_200.tsv"

OUTPUT_DIR = "llama/fewshot"
PRED_FILE = f"{OUTPUT_DIR}/pred_urd.csv"
EVAL_FILE = f"{OUTPUT_DIR}/eval_urd.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}

LABEL_LIST = ["Political", "Racial/Ethnic", "Religious", "Gender/Sexual", "Other"]


# =========================================================
# BINARY LABEL ENCODING
# =========================================================
def to_binary(lbls):
    if "None" in lbls:
        return [0, 0, 0, 0, 0]
    return [1 if x in lbls else 0 for x in LABEL_LIST]


# =========================================================
# LOAD TRAIN SET, FORM FEW-SHOT EXAMPLES
# =========================================================
df_full = pd.read_csv(TRAIN_FILE, sep="\t")

buckets = {
    "Political": [],
    "Racial/Ethnic": [],
    "Religious": [],
    "Gender/Sexual": [],
    "Other": [],
    "None": []
}

for _, row in df_full.iterrows():
    lbl = row["labels"]

    if not isinstance(lbl, str) or lbl.strip() == "" or lbl.lower() == "none":
        buckets["None"].append(row)
        continue

    parts = [x.strip() for x in lbl.split(",")]
    for p in parts:
        if p in buckets:
            buckets[p].append(row)

# Urdu strongly multi-label → choose rich examples
sample_plan = {
    "Political": 2,
    "Racial/Ethnic": 2,
    "Religious": 2,
    "Gender/Sexual": 2,
    "Other": 2,
    "None": 2
}

fewshot_examples = []
for label, count in sample_plan.items():
    if len(buckets[label]) >= count:
        fewshot_examples.extend(random.sample(buckets[label], count))
    else:
        fewshot_examples.extend(buckets[label])


# =========================================================
# BUILD FEW-SHOT TEXT BLOCK
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


# =========================================================
# URDU SYSTEM PROMPT
# =========================================================
SYSTEM_PROMPT = f"""
You are an expert analyst of Urdu social-media discourse.
Your task is SemEval 2026 Task 9 — Polarization Type Classification (multi-label).

Urdu dataset characteristics:
• Very high overlap between political, religious, racial/ethnic, and gender hostility.
• Sectarian references are common: شیعہ، سنی، دیوبندی، بریلوی.
• National/ethnic references: پاکستانی، ہندوستانی، پٹھان، بنگالی، سندھی.
• Political figures and parties used aggressively.
• Gender and sexuality terms appear in insults: عورت دشمنی، مردانہ تعصب، “خواجہ سرا”.
• Multi-label hostility is extremely common — often 2–4 labels per text.
• Sarcasm and metaphors occur but must show explicit group targeting.

VALID LABELS:
Political, Racial/Ethnic, Religious, Gender/Sexual, Other, None

DEFINITIONS:
• Political = hostility against political groups, parties, supporters, ideologies.
• Racial/Ethnic = ethnicity/nationality-based hostility (پٹھان، سندھی، بنگالی، بھارتی).
• Religious = sectarian or faith-based hostility (شیعہ، سنی، ہندو، عیسائی).
• Gender/Sexual = misogyny, homophobia, attacks on women or LGBTQ groups.
• Other = hostility toward any other identifiable group.
• None = no explicit group-based hostility.

RULES:
• Hostility MUST be directed at a group, not an individual.
• Strong multi-label cases are common — mark ALL relevant labels.
• Do NOT over-label: text must clearly target a group.
• Emojis NEVER imply hostility by themselves.

FEW-SHOT EXAMPLES:
{fewshot_text}

Return ONLY JSON like:
{{"labels": "Political"}}
{{"labels": "Political, Religious"}}
{{"labels": "None"}}
"""


# =========================================================
# LLAMA CALLER
# =========================================================
def call_llama(text, max_retries=5):
    user_prompt = (
        f'Text: "{text}"\n'
        f'Return JSON only:\n'
        f'{{"labels": "..."}}'
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 128,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
    }

    delay = 1.0
    for _ in range(max_retries):
        try:
            resp = requests.post(invoke_url, headers=HEADERS, json=payload)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
        except Exception:
            traceback.print_exc()

        print(f"↻ Retrying in {delay}s...")
        time.sleep(delay)
        delay *= 2

    return None


# =========================================================
# PARSE OUTPUT
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
# MAIN
# =========================================================
def main():
    print("\n=== FEW-SHOT LLAMA Classification (Urdu / urd) ===\n")

    df = pd.read_csv(SUBSET_FILE, sep="\t")

    preds = []
    gold = []

    for idx, row in df.iterrows():
        text = row["text"]
        gold_lbl = row["labels"]

        if not isinstance(gold_lbl, str) or gold_lbl.strip() == "" or gold_lbl.lower() == "none":
            gold_list = ["None"]
        else:
            gold_list = [x.strip() for x in gold_lbl.split(",")]

        gold.append(to_binary(gold_list))

        raw = call_llama(text)
        parsed = parse_labels(raw)
        preds.append(to_binary(parsed))

        print(f"[{idx+1}/{len(df)}] {row['id']} → {parsed}")
        time.sleep(0.1)

    # Save predictions
    pred_df = pd.DataFrame(preds, columns=[
        "political", "racial/ethnic", "religious", "gender/sexual", "other"
    ])
    pred_df.insert(0, "id", df["id"])
    pred_df.to_csv(PRED_FILE, index=False)

    print("\nSaved predictions →", PRED_FILE)

    # Evaluation
    gold_df = pd.DataFrame(gold, columns=pred_df.columns[1:])
    pred_vals = pred_df.drop(columns=["id"])

    lines = ["=== CLASSWISE F1 SCORES ==="]
    for col in pred_vals.columns:
        f1 = f1_score(gold_df[col], pred_vals[col], zero_division=0)
        lines.append(f"{col}: {f1:.4f}")

    macro = f1_score(gold_df, pred_vals, average="macro", zero_division=0)
    lines.append(f"\nMacro F1: {macro:.4f}")

    with open(EVAL_FILE, "w") as f:
        f.write("\n".join(lines))

    print("\nSaved evaluation →", EVAL_FILE)
    print("\n=== DONE (FEW-SHOT Urdu) ===\n")


if __name__ == "__main__":
    main()

