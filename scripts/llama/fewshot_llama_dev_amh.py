import os
import time
import json
import random
import requests
import pandas as pd


# ============================================================
#                   CONFIG & PATHS
# ============================================================

DEV_FILE = "Dataset/dev_phase/subtask2/dev/amh.csv"
TRAIN_FILE = "Dataset/Processed/amh.tsv"

OUT_DIR = "llama/dev/fewshot"
os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_FILE = f"{OUT_DIR}/amh.csv"

API_KEY = os.environ.get("NVIDIA_API_KEY")
if API_KEY is None:
    raise ValueError("ERROR: NVIDIA_API_KEY is not set!")

INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

LABELS = ["Political", "Racial/Ethnic", "Religious", "Gender/Sexual", "Other"]

MAX_EXAMPLES = 10  # few-shot examples count


# ============================================================
#          LOAD RANDOM FEW-SHOT EXAMPLES FROM TRAIN SET
# ============================================================

def load_fewshot_examples():
    df = pd.read_csv(TRAIN_FILE, sep="\t")

    # keep only labeled examples
    df = df[df["labels"] != "None"]

    # pick random 10 for few-shot
    examples = df.sample(min(MAX_EXAMPLES, len(df)), random_state=42)

    txt = ""
    for _, row in examples.iterrows():
        t = row["text"]
        l = row["labels"]
        txt += f"Text: {t}\nLabels: {l}\n\n"

    return txt.strip()


fewshot_text = load_fewshot_examples()


# ============================================================
#                SYSTEM PROMPT (EXACT FORMAT GIVEN)
# ============================================================

SYSTEM_PROMPT = f"""
You are an expert analyst of Amharic social media discourse. You classify short Amharic posts 
for SemEval 2026 Task 9 (Subtask 2): Polarization Type Classification.

Amharic content often involves political conflict, ethnic tension, religious identity, and 
sarcastic rhetorical attacks. Your task is to detect ONLY explicit polarization directed at 
identifiable social or political groups.

VALID LABELS:
Political, Racial/Ethnic, Religious, Gender/Sexual, Other, None

DEFINITIONS:
• Political: criticism or hostility toward political leaders, government, parties, 
  movements, or ideology.
• Racial/Ethnic: hostility toward ethnic groups (Amhara, Oromo, Tigray, Somali, etc.).
• Religious: hostility toward religious groups, clergy, or beliefs.
• Gender/Sexual: misogyny, gender harassment, LGBTQ+ hostility.
• Other: any other group-based hostility not covered above.
• If the text contains no explicit hostility, output: None.

IMPORTANT:
• Do NOT infer hidden meaning; classify only explicit group-directed hostility.
• Amharic posts often use irony or rhetorical style; this counts only if clearly directed 
  toward a group.
• Multi-label outputs are allowed when multiple groups are targeted.
• Emojis alone do not indicate hostility.

Below are few-shot examples:

{fewshot_text}

Now classify the following text. Return ONLY this format:
{{"labels": "<comma-separated list or None>"}}
"""


# ============================================================
#            USER PROMPT BUILDER (NO EXTRA TEXT)
# ============================================================

def build_user_prompt(text):
    return f'Text: "{text}"'


# ============================================================
#             NVIDIA NIM LLAMA API CALL + RETRY
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

            print(f"[WARN] Status {r.status_code}: {r.text}")
            time.sleep(sleep * (attempt + 1))

        except Exception as e:
            print(f"[ERROR] {e} — retrying...")
            time.sleep(sleep * (attempt + 1))

    return '{"labels": "None"}'


# ============================================================
#         PARSE MODEL OUTPUT → BINARY VECTOR OF 5 LABELS
# ============================================================

def decode_labels(json_text):
    """
    Expected format:
        {"labels": "Political, Religious"}
    or:
        {"labels": "None"}
    """
    try:
        obj = json.loads(json_text)
        label_str = obj.get("labels", "None")

        if label_str is None or label_str.strip() == "" or label_str == "None":
            return ["None"]

        labels = [x.strip() for x in label_str.split(",")]
        return labels

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
    print("\n=== FEW-SHOT LLAMA DEV INFERENCE (AMH) ===")

    df = pd.read_csv(DEV_FILE)

    # remove empty gold columns if included in the CSV
    drop_cols = ["political", "racial/ethnic", "religious", "gender/sexual", "other"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    preds = []

    for i, row in df.iterrows():
        text_id = row["id"]
        text = str(row["text"])

        print(f"[{i+1}/{len(df)}] ID={text_id}")

        prompt = build_user_prompt(text)
        raw = call_llama(prompt)

        parsed_labels = decode_labels(raw)
        binary = labels_to_binary(parsed_labels)

        preds.append([text_id] + binary)

        time.sleep(0.15)

    # save predictions
    out_df = pd.DataFrame(preds, columns=["id"] + LABELS)
    out_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\nSaved predictions → {OUTPUT_FILE}")
    print("=== DONE ===\n")


if __name__ == "__main__":
    main()

