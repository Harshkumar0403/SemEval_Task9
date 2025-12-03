import os
import time
import json
import random
import requests
import pandas as pd


# ============================================================
#                        CONFIG
# ============================================================

DEV_FILE = "Dataset/dev_phase/subtask2/dev/ita.csv"
TRAIN_FILE = "Dataset/Processed/ita.tsv"

OUT_DIR = "llama/dev/fewshot"
os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_FILE = f"{OUT_DIR}/ita.csv"

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

    # keep only labeled examples
    df = df[df["labels"] != "None"]

    examples = df.sample(min(MAX_EXAMPLES, len(df)), random_state=42)

    txt = ""
    for _, row in examples.iterrows():
        t = row["text"]
        lbl = row["labels"]
        txt += f"Text: {t}\nLabels: {lbl}\n\n"

    return txt.strip()


fewshot_text = load_fewshot_examples()


# ============================================================
#           SYSTEM PROMPT (ITALIAN FEWSHOT VERSION)
# ============================================================

SYSTEM_PROMPT = f"""
You are an expert analyst of Italian social media discourse. You classify short posts 
for SemEval 2026 Task 9 (Subtask 2): Polarization Type Classification.

Italian social media often contains:
• political hostility (government, parties, ideologies),
• racial/ethnic tension (immigration, minority groups, stereotypes),
• religion-based hostility (Christianity, Islam, clergy, cultural symbols),
• gender-based harassment, misogyny, LGBTQ+ hostility,
• sarcasm, humor, and heavy emoji usage.

Your task is to detect ONLY explicit group-directed polarization.

VALID LABELS:
Political, Racial/Ethnic, Religious, Gender/Sexual, Other, None

DEFINITIONS:
• Political — hostility toward political parties, leaders, ideology.
• Racial/Ethnic — xenophobia, anti-immigrant sentiment, minority stereotyping.
• Religious — insults toward religious groups, clergy, or faith symbols.
• Gender/Sexual — misogyny, homophobia, transphobia.
• Other — group hostility not covered above.
• None — if the post shows no explicit hostility to groups.

IMPORTANT:
• Do NOT infer hidden meaning; classify only explicit hostility.
• Italian posts may use sarcasm or rhetorical style; count only if hostility is clear.
• Multi-label outputs are allowed.
• Emojis alone do NOT imply hostility.

Few-shot examples:

{fewshot_text}

Now classify the following text. Return ONLY in this JSON format:
{{"labels": "<comma-separated list or None>"}}
"""


# ============================================================
#                   USER PROMPT BUILDER
# ============================================================

def build_user_prompt(text):
    return f'Text: "{text}"'


# ============================================================
#         LLAMA API CALL WITH RETRY & BACKOFF
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
#         PARSE MODEL OUTPUT → LABEL LIST
# ============================================================

def decode_labels(json_text):
    try:
        obj = json.loads(json_text)
        label_str = obj.get("labels", "None")

        if not label_str or label_str == "None":
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
    print("\n=== FEW-SHOT LLAMA DEV INFERENCE (ITALIAN / ITA) ===")

    df = pd.read_csv(DEV_FILE)

    # Dev set has empty columns → remove all gold labels
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

        time.sleep(0.15)

    out_df = pd.DataFrame(preds, columns=["id"] + LABELS)
    out_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\nSaved predictions → {OUTPUT_FILE}")
    print("=== DONE ===\n")


if __name__ == "__main__":
    main()

