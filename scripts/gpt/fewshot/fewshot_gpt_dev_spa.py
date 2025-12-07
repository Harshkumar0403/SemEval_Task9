import os
import time
import json
import re
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ============================================================
#                CONFIG
# ============================================================
LANG = "spa"
TRAIN_FILE = f"Dataset/Processed/{LANG}.tsv"
DEV_FILE = f"Dataset/dev_phase/subtask2/dev/{LANG}.csv"
OUTPUT_DIR = "gpt/dev/fewshot"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_EXAMPLES = 10
SLEEP_TIME = 0.01  # pacing to avoid hitting rate limits

LABELS = ["Political", "Racial/Ethnic", "Religious", "Gender/Sexual", "Other"]


# ============================================================
#           API KEY
# ============================================================
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("OPENAI_API_KEY not set!")

client = OpenAI(api_key=api_key)


# ============================================================
#        LOAD FEW-SHOT EXAMPLES
# ============================================================
def load_fewshot_examples():
    df = pd.read_csv(TRAIN_FILE, sep="\t")
    df = df[df["labels"] != "None"]  # remove unlabeled

    examples = df.sample(min(MAX_EXAMPLES, len(df)), random_state=42)

    txt = ""
    for _, row in examples.iterrows():
        txt += f"Text: {row['text']}\nLabels: {row['labels']}\n\n"

    return txt.strip()


fewshot_text = load_fewshot_examples()


# ============================================================
#               SYSTEM PROMPT (Spanish)
# ============================================================
SYSTEM_PROMPT = f"""
You are an expert analyst of Spanish social-media discourse. Your task is to classify each post
into ANY number of the following categories (multi-label allowed):

1. Political
2. Racial/Ethnic
3. Religious
4. Gender/Sexual
5. Other
6. None — ONLY if no harmful or targeted content exists.

Rules:
- Output MUST be a JSON list, e.g. ["Political", "Racial/Ethnic"].
- If nothing applies, return ["None"].
- Do NOT output explanations or prose.

Guidelines:
Spanish posts often include political insults, xenophobic remarks, anti-religious sentiment,
gender-based harassment, and other targeted hostility. Label ONLY explicit hostility toward
groups, not general negativity or jokes.

Few-shot examples:

{fewshot_text}

Now classify the following text. Return ONLY this format:
{{"labels": "<comma-separated list or None>"}}
"""


# ============================================================
#   ROBUST LABEL PARSER
# ============================================================
def extract_labels(text):
    # Try JSON first
    try:
        obj = json.loads(text)
        if "labels" in obj:
            raw = obj["labels"]
            if raw.lower() == "none":
                return []
            return [x.strip() for x in raw.split(",") if x.strip()]
    except:
        pass

    # Regex fallback
    m = re.search(r'{"labels":\s*"(.*?)"}', text)
    if m:
        raw = m.group(1)
        if raw.lower() == "none":
            return []
        return [x.strip() for x in raw.split(",")]

    return []


# ============================================================
#                GPT CALL
# ============================================================
def gpt_predict(text):
    for _ in range(5):
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                max_tokens=48,
            )

            content = response.choices[0].message.content
            return extract_labels(content)

        except Exception as e:
            print("Retry due to error:", e)
            time.sleep(1)

    return []


# ============================================================
#         CONVERT LABELS → MULTI-HOT
# ============================================================
def labels_to_binary(pred_list):
    vec = [0] * len(LABELS)
    for p in pred_list:
        if p in LABELS:
            vec[LABELS.index(p)] = 1
    return vec


# ============================================================
#                       MAIN LOOP
# ============================================================
def run():
    print(f"\n=== GPT-4.1 Few-shot inference for {LANG} ===\n")

    df = pd.read_csv(DEV_FILE)
    out_path = f"{OUTPUT_DIR}/pred_{LANG}.csv"

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        uid = row["id"]
        text = str(row["text"])

        labels = gpt_predict(text)
        vec = labels_to_binary(labels)

        rows.append([uid] + vec)
        time.sleep(SLEEP_TIME)

    out_df = pd.DataFrame(rows, columns=["id"] + LABELS)
    out_df.to_csv(out_path, index=False)

    print(f"\n✔ Saved → {out_path}\n")


if __name__ == "__main__":
    run()

