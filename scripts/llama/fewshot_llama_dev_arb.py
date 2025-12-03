import os
import time
import json
import random
import requests
import pandas as pd


# ============================================================
#                   CONFIG & PATHS
# ============================================================

DEV_FILE = "Dataset/dev_phase/subtask2/dev/arb.csv"
TRAIN_FILE = "Dataset/Processed/arb.tsv"

OUT_DIR = "llama/dev/fewshot"
os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_FILE = f"{OUT_DIR}/arb.csv"

API_KEY = os.environ.get("NVIDIA_API_KEY")
if API_KEY is None:
    raise ValueError("ERROR: NVIDIA_API_KEY is not set!")

INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

LABELS = ["Political", "Racial/Ethnic", "Religious", "Gender/Sexual", "Other"]

MAX_EXAMPLES = 10  # few-shot size


# ============================================================
#     LOAD FEW-SHOT EXAMPLES FROM TRAIN SET (random 10)
# ============================================================

def load_fewshot_examples():
    df = pd.read_csv(TRAIN_FILE, sep="\t")

    df = df[df["labels"] != "None"]  # keep only labeled examples
    examples = df.sample(min(MAX_EXAMPLES, len(df)), random_state=42)

    text = ""
    for _, row in examples.iterrows():
        t = row["text"]
        l = row["labels"]
        text += f"Text: {t}\nLabels: {l}\n\n"

    return text.strip()


fewshot_text = load_fewshot_examples()


# ============================================================
#            SYSTEM PROMPT FOR ARABIC (MATCH TRAINING)
# ============================================================

SYSTEM_PROMPT = f"""
You are an expert analyst of Arabic social media discourse. You classify short Arabic posts 
for SemEval 2026 Task 9 (Subtask 2): Polarization Type Classification.

Arabic content often includes political sarcasm, sectarian tension, ethnic references, gender 
harassment, religious identity, and rhetorical attacks. Your task is to detect ONLY explicit 
polarization directed at social, religious, political, or identity-based groups.

VALID LABELS:
Political, Racial/Ethnic, Religious, Gender/Sexual, Other, None

DEFINITIONS:
• Political: hostility toward political figures, parties, governments, policies, or movements.
• Racial/Ethnic: hostility toward Arabs, Kurds, Amazigh, Nubians, Africans, etc.
• Religious: hostility toward Sunni, Shia, Christian, Jewish, or any religious group.
• Gender/Sexual: misogyny, harassment, anti-LGBTQ hostility, gender insults.
• Other: any group-based hostility not covered above.
• Use None ONLY when no explicit hostility exists.

IMPORTANT:
• Do NOT infer hidden intentions. Classify only explicit hostility toward groups.
• Arabic posts may use insults, emojis, sarcasm, or coded references — classify only if they 
  explicitly target a group.
• A post may have multiple labels.
• Emojis alone do NOT indicate polarization.

Few-shot examples:

{fewshot_text}

Now classify the following text. Return ONLY this format:
{{"labels": "<comma-separated list or None>"}}
"""


# ============================================================
#                USER PROMPT BUILDER
# ============================================================

def build_user_prompt(text):
    return f'Text: "{text}"'


# ============================================================
#              NVIDIA LLAMA API CALL + RETRY
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
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
    }

    for attempt in range(retries):
        try:
            r = requests.post(INVOKE_URL, headers=headers, json=payload, timeout=60)

            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]

            print(f"[WARN] HTTP {r.status_code}: {r.text}")
            time.sleep(sleep * (attempt + 1))

        except Exception as e:
            print(f"[ERROR] {e} — retrying...")
            time.sleep(sleep * (attempt + 1))

    return '{"labels": "None"}'


# ============================================================
#           PARSE MODEL OUTPUT → LIST OF LABELS
# ============================================================

def decode_labels(json_text):
    try:
        obj = json.loads(json_text)
        label_str = obj.get("labels", "None")

        if label_str is None or label_str.strip() == "" or label_str == "None":
            return ["None"]

        return [x.strip() for x in label_str.split(",")]

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
    print("\n=== FEW-SHOT LLAMA DEV INFERENCE (ARABIC / ARB) ===")

    df = pd.read_csv(DEV_FILE)

    # Remove gold-label columns if present
    cols_to_remove = ["political", "racial/ethnic", "religious", "gender/sexual", "other"]
    df = df.drop(columns=[c for c in cols_to_remove if c in df.columns])

    preds = []

    for i, row in df.iterrows():
        text_id = row["id"]
        text = str(row["text"])

        print(f"[{i+1}/{len(df)}] ID={text_id}")

        raw = call_llama(build_user_prompt(text))
        parsed = decode_labels(raw)
        binary = labels_to_binary(parsed)

        preds.append([text_id] + binary)

        time.sleep(0.15)

    out_df = pd.DataFrame(preds, columns=["id"] + LABELS)
    out_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\nSaved predictions → {OUTPUT_FILE}")
    print("=== DONE ===\n")


if __name__ == "__main__":
    main()

