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
LANG = "hau"
TRAIN_FILE = f"Dataset/Processed/{LANG}.tsv"
DEV_FILE = f"Dataset/dev_phase/subtask2/dev/{LANG}.csv"
OUTPUT_DIR = "deepseek/dev/fewshot"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_EXAMPLES = 10
SLEEP_TIME = 0.01  # 100 rpm pacing

LABELS = ["Political", "Racial/Ethnic", "Religious", "Gender/Sexual", "Other"]


# ============================================================
#                API KEY
# ============================================================
api_key = os.getenv("DEEPSEEK_API_KEY")
if api_key is None:
    raise ValueError("DEEPSEEK_API_KEY not set!")

client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")


# ============================================================
#            FEW-SHOT EXAMPLE LOADER
# ============================================================
def load_fewshot_examples():
    df = pd.read_csv(TRAIN_FILE, sep="\t")
    df = df[df["labels"] != "None"]

    examples = df.sample(min(MAX_EXAMPLES, len(df)), random_state=42)

    txt = ""
    for _, row in examples.iterrows():
        t = row["text"]
        l = row["labels"]
        txt += f"Text: {t}\nLabels: {l}\n\n"

    return txt.strip()


fewshot_text = load_fewshot_examples()


# ============================================================
#        SYSTEM PROMPT (HAUSA FEWSHOT)
# ============================================================
SYSTEM_PROMPT = f"""
You are an expert analyst of Hausa social-media discourse. Your task is to detect hateful,
abusive, or targeted content and classify each post into ANY number of the categories below
(multi-label allowed):

1. Political
2. Racial/Ethnic
3. Religious
4. Gender/Sexual
5. Other
6. None — ONLY if there is no harmful or targeted content.

Rules:
- Output MUST be a JSON list, for example: ["Political", "Religious"].
- No explanations. No prose.
- If no category applies, return ["None"].

Guidance:
Hausa social-media content often includes political commentary (government, parties,
leaders), ethnic or regional tensions, religious criticism (Islamic groups, clerics),
and gender-based insults. Classify ONLY explicit hostility or group targeting.

Few-shot examples:

{fewshot_text}

Now classify the following text. Return ONLY:
{{"labels": "<comma-separated list or None>"}}
"""


# ============================================================
#          JSON LABEL EXTRACTION (robust)
# ============================================================
def extract_labels(text):
    # Try parsing as JSON
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
    match = re.search(r'{"labels":\s*"(.*?)"}', text)
    if match:
        raw = match.group(1)
        if raw.lower() == "none":
            return []
        return [x.strip() for x in raw.split(",")]

    return []


# ============================================================
#                   GPT Predict (retry)
# ============================================================
def gpt_predict(text):
    for _ in range(5):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
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
#        Convert label list → binary vector
# ============================================================
def labels_to_binary(pred_list):
    vec = [0] * len(LABELS)
    for p in pred_list:
        if p in LABELS:
            vec[LABELS.index(p)] = 1
    return vec


# ============================================================
#                      MAIN LOOP
# ============================================================
def run():
    print(f"\n=== Deepseek FEWSHOT inference for {LANG} ===\n")

    df = pd.read_csv(DEV_FILE)
    out_path = f"{OUTPUT_DIR}/pred_{LANG}.csv"

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        uid = row["id"]
        text = str(row["text"])

        labels = gpt_predict(text)
        binary = labels_to_binary(labels)

        rows.append([uid] + binary)
        time.sleep(SLEEP_TIME)

    out_df = pd.DataFrame(rows, columns=["id"] + LABELS)
    out_df.to_csv(out_path, index=False)

    print(f"\n✔ Saved → {out_path}\n")


if __name__ == "__main__":
    run()

