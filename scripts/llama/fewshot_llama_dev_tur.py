import os
import time
import json
import random
import requests
import pandas as pd


# ============================================================
#                        CONFIG
# ============================================================

DEV_FILE = "Dataset/dev_phase/subtask2/dev/tur.csv"
TRAIN_FILE = "Dataset/Processed/tur.tsv"

OUT_DIR = "llama/dev/fewshot"
os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_FILE = f"{OUT_DIR}/tur.csv"

API_KEY = os.environ.get("NVIDIA_API_KEY")
if API_KEY is None:
    raise ValueError("ERROR: NVIDIA_API_KEY is not set!")

INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

LABELS = ["Political", "Racial/Ethnic", "Religious", "Gender/Sexual", "Other"]
MAX_EXAMPLES = 10


# ============================================================
#      LOAD RANDOM FEWSHOT EXAMPLES FROM TRAIN SET
# ============================================================

def load_fewshot_examples():
    df = pd.read_csv(TRAIN_FILE, sep="\t")

    df = df[df["labels"] != "None"]  # only keep labeled rows

    examples = df.sample(min(MAX_EXAMPLES, len(df)), random_state=42)

    text_block = ""
    for _, row in examples.iterrows():
        t = row["text"]
        lbl = row["labels"]
        text_block += f"Text: {t}\nLabels: {lbl}\n\n"

    return text_block.strip()


fewshot_text = load_fewshot_examples()


# ============================================================
#          TURKISH SYSTEM PROMPT (FEWSHOT VERSION)
# ============================================================

SYSTEM_PROMPT = f"""
You are an expert analyst of Turkish social media discourse.
Your task is SemEval 2026 Task 9 (Subtask 2): Polarization Type Classification.

Turkish online conversations frequently involve political debate, ethnic identity
conflict (e.g., Türk, Kürt, Suriyeli), religious criticism, and gender/sexual hostility.
You must detect ONLY explicit group-directed hostility.

VALID LABELS:
Political, Racial/Ethnic, Religious, Gender/Sexual, Other, None

DEFINITIONS:
• Political — hostility toward political leaders, parties, ideologies, or government institutions.
• Racial/Ethnic — hostility toward ethnic groups (Türk, Kürt, Arap, Ermeni, Suriyeli, etc.).
• Religious — hostility toward religious groups, beliefs, clerics, or institutions.
• Gender/Sexual — misogyny, harassment towards women, LGBTQ+ hostility.
• Other — identifiable group hostility not covered above.
• None — no explicit hostility.

IMPORTANT RULES:
• Do NOT infer hidden meaning; classify only explicit hostility.
• Sarcasm counts only if clearly directed at a group.
• Multi-label allowed.
• Emojis alone do not imply hostility.

Few-shot examples below:

{fewshot_text}

Now classify the following text. Return ONLY this JSON:
{{"labels": "<comma-separated list or None>"}}
"""


# ============================================================
#                    USER PROMPT BUILDER
# ============================================================

def build_user_prompt(text):
    return f'Text: "{text}"'


# ============================================================
#          LLAMA API CALL WITH RETRY & BACKOFF
# ============================================================

def call_llama(prompt, retries=5, sleep=0.25):
    payload = {
        "model": "meta/llama-4-maverick-17b-128e-instruct",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.15,
        "max_tokens": 128,
        "top_p": 1.0,
        "stream": False
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json"
    }

    for attempt in range(retries):
        try:
            r = requests.post(INVOKE_URL, headers=headers, json=payload, timeout=60)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            print(f"[WARN] HTTP {r.status_code}: {r.text}")
            time.sleep(sleep * (attempt + 1))
        except Exception as e:
            print(f"[ERROR] {e} — retrying…")
            time.sleep(sleep * (attempt + 1))

    return '{"labels": "None"}'


# ============================================================
#             PARSE MODEL OUTPUT → LABEL LIST
# ============================================================

def decode_labels(out_text):
    try:
        obj = json.loads(out_text)
        lbl_str = obj.get("labels", "None")

        if not lbl_str or lbl_str == "None":
            return ["None"]

        return [x.strip() for x in lbl_str.split(",")]

    except Exception:
        return ["None"]


def labels_to_binary(labels):
    if "None" in labels:
        return [0, 0, 0, 0, 0]
    return [1 if lbl in labels else 0 for lbl in LABELS]


# ============================================================
#                           MAIN
# ============================================================

def main():
    print("\n=== FEW-SHOT LLAMA DEV INFERENCE (TURKISH / TUR) ===")

    df = pd.read_csv(DEV_FILE)

    # Dev set has empty gold columns → remove them
    for col in ["political", "racial/ethnic", "religious", "gender/sexual", "other"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    preds = []

    for i, row in df.iterrows():
        text_id = row["id"]
        text = str(row["text"])

        print(f"[{i+1}/{len(df)}] ID={text_id}")

        raw = call_llama(build_user_prompt(text))
        parsed = decode_labels(raw)
        binary = labels_to_binary(parsed)

        preds.append([text_id] + binary)

        time.sleep(0.15)  # rate control

    out_df = pd.DataFrame(preds, columns=["id"] + LABELS)
    out_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\nSaved predictions → {OUTPUT_FILE}")
    print("=== DONE ===\n")


if __name__ == "__main__":
    main()

