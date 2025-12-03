import os
import time
import json
import random
import requests
import pandas as pd


# ============================================================
#                        CONFIG
# ============================================================

DEV_FILE = "Dataset/dev_phase/subtask2/dev/hin.csv"
TRAIN_FILE = "Dataset/Processed/hin.tsv"

OUT_DIR = "llama/dev/fewshot"
os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_FILE = f"{OUT_DIR}/hin.csv"

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

    df = df[df["labels"] != "None"]  # keep only labeled examples
    examples = df.sample(min(MAX_EXAMPLES, len(df)), random_state=42)

    txt = ""
    for _, row in examples.iterrows():
        t = row["text"]
        l = row["labels"]
        txt += f"Text: {t}\nLabels: {l}\n\n"

    return txt.strip()


fewshot_text = load_fewshot_examples()


# ============================================================
#              SYSTEM PROMPT FOR HINDI (MATCH TRAINING)
# ============================================================

SYSTEM_PROMPT = f"""
You are an expert analyst of Hindi and Hinglish social media discourse. You classify short posts 
for SemEval 2026 Task 9 (Subtask 2): Polarization Type Classification.

Hindi social media often contains:
• political hostility and party-based insults (BJP, Congress, Modi, Yogi, etc.),
• caste-based hostility and ethnic slurs (Dalit, Brahmin, OBC, SC/ST),
• religious polarization (Hindu–Muslim tension, temple–mosque disputes),
• gender-based harassment, sexism, and LGBTQ hostility,
• Hinglish slang, sarcasm, and abusive humor,
• heavy emoji usage.

Your task is to detect ONLY explicit group-directed polarization.

VALID LABELS:
Political, Racial/Ethnic, Religious, Gender/Sexual, Other, None

DEFINITIONS:
• Political — hostility toward Indian political parties, leaders, governments, ideology.
• Racial/Ethnic — caste insults, community stereotyping, anti-regional slurs.
• Religious — Hindu–Muslim tension, attacks on religious groups, identity mocking.
• Gender/Sexual — misogyny, homophobia, transphobia, gender-based attacks.
• Other — any group-based hostility not covered above.
• None — if no explicit hostility appears.

IMPORTANT:
• Classify ONLY when there is explicit hostility toward an identifiable group.
• Do NOT infer hidden meaning from sarcasm unless the target group is clear.
• Hinglish vocabulary ("bhakt", "liberandu", "mullah", "andhbhakt") often indicates hostility — label only if directed at a group.
• Multiple labels may apply.
• Emojis alone do NOT indicate hostility.

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

        if label_str is None or label_str.strip() in ("", "None"):
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
    print("\n=== FEW-SHOT LLAMA DEV INFERENCE (HINDI / HIN) ===")

    df = pd.read_csv(DEV_FILE)

    # Remove gold columns (all blank in dev)
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

