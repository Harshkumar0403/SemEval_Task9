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

TRAIN_FILE = "Dataset/Processed/tur.tsv"
SUBSET_FILE = "Dataset/Subset/tur_200.tsv"

OUTPUT_DIR = "llama/fewshot"
PRED_FILE = f"{OUTPUT_DIR}/pred_tur.csv"
EVAL_FILE = f"{OUTPUT_DIR}/eval_tur.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}

LABEL_LIST = ["Political", "Racial/Ethnic", "Religious", "Gender/Sexual", "Other"]


# =========================================================
# BINARY LABEL ENCODER
# =========================================================
def to_binary(lbls):
    if "None" in lbls:
        return [0, 0, 0, 0, 0]
    return [1 if x in lbls else 0 for x in LABEL_LIST]


# =========================================================
# LOAD TRAIN SET → FEW-SHOT SAMPLING
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

# Balanced sample allocation for Turkish
sample_plan = {
    "Political": 3,
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
# FORMAT FEW-SHOT BLOCK
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
# SYSTEM PROMPT FOR TURKISH (tur)
# =========================================================
SYSTEM_PROMPT = f"""
You are an expert analyst of Turkish social-media discourse.
Task: SemEval 2026 Task 9 — Polarization Type Classification (multi-label).

Turkish dataset characteristics:
• Very high political hostility (AKP, CHP, iktidar, muhalefet, seçim, Erdoğan).
• Racial/Ethnic tension referencing groups such as Kürtler, Türkler, Ermeniler, Suriyeliler.
• Religious polarization involving laik–dindar tensions, Alevi–Sünni references.
• Gender/Sexual hostility including misogyny and LGBTQ-phobic language.
• Sarcasm and metaphors are common, but DO NOT infer hostility unless explicit.
• Emojis like 😂🤣🤦 do NOT imply aggression by themselves.

VALID LABELS:
Political, Racial/Ethnic, Religious, Gender/Sexual, Other, None

DEFINITIONS:
• Political = hostility toward political groups, supporters, ideologies, parties.
• Racial/Ethnic = hostility toward groups based on ethnicity, nationality, or shared heritage.
• Religious = hostility toward Islamic groups, Alevi/Sunni groups, secular/religious identity.
• Gender/Sexual = attacks on women, men, LGBTQ people, feminists.
• Other = group-based hostility not covered above.
• None = no explicit hostility.

RULES:
• Only label explicit hostility — NO guessing.
• Group must be identifiable (e.g., “Suriyeliler”, “AKP’liler”).
• Multi-label allowed.
• If uncertain, choose “None”.

FEW-SHOT EXAMPLES:
{fewshot_text}

Return ONLY JSON, e.g.:
{{"labels": "Political"}}
{{"labels": "Political, Racial/Ethnic"}}
{{"labels": "None"}}
"""


# =========================================================
# CALL LLAMA API
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
# PARSE MODEL OUTPUT
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
# MAIN INFERENCE LOOP
# =========================================================
def main():
    print("\n=== FEW-SHOT LLAMA Classification (Turkish / tur) ===\n")

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
    print("\n=== DONE (FEW-SHOT Turkish) ===\n")


if __name__ == "__main__":
    main()

