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
LANG = "hin"
TRAIN_FILE = f"Dataset/Processed/{LANG}.tsv"
DEV_FILE = f"Dataset/dev_phase/subtask2/dev/{LANG}.csv"
OUTPUT_DIR = "deepseek/dev/fewshot"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_EXAMPLES = 10
SLEEP_TIME = 0.01  # pacing for 100 rpm

LABELS = ["Political", "Racial/Ethnic", "Religious", "Gender/Sexual", "Other"]


# ============================================================
#                API KEY
# ============================================================
api_key = os.getenv("DEEPSEEK_API_KEY")
if api_key is None:
    raise ValueError("DEEPSEEK_API_KEY not set!")

client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")


# ============================================================
#            FEW-SHOT EXAMPLES LOADER
# ============================================================
def load_fewshot_examples():
    df = pd.read_csv(TRAIN_FILE, sep="\t")
    df = df[df["labels"] != "None"]  # keep only labeled samples

    examples = df.sample(min(MAX_EXAMPLES, len(df)), random_state=42)

    txt = ""
    for _, row in examples.iterrows():
        t = row["text"]
        l = row["labels"]
        txt += f"Text: {t}\nLabels: {l}\n\n"

    return txt.strip()


fewshot_text = load_fewshot_examples()


# ============================================================
#            SYSTEM PROMPT (Hindi Few-shot)
# ============================================================
SYSTEM_PROMPT = f"""
You are an expert analyst of Hindi social-media discourse. Your task is to detect hateful,
abusive, or targeted content and classify each post into ANY number of the following categories
(multi-label classification):

1. Political
2. Racial/Ethnic
3. Religious
4. Gender/Sexual
5. Other
6. None — ONLY if the text contains no harmful or targeted content.

Rules:
- Output MUST be a JSON list, e.g.: ["Political", "Religious"].
- No explanations, no prose.
- If nothing applies, output ["None"].

Guidance:
Hindi social-media posts often involve political criticism (leaders, parties, ideology),
religious tensions, caste-based hostility, gender-based abuse, and community targeting.
Classify ONLY explicit hostility toward identifiable groups.

Few-shot examples:

{fewshot_text}

Now classify the following text. Return ONLY:
{{"labels": "<comma-separated list or None>"}}
"""


# ============================================================
#        JSON LABEL PARSING (robust)
# ============================================================
def extract_labels(text):
    # Try JSON
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
#                 GPT Predict
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
#         Convert labels → multi-hot vector
# ============================================================
def labels_to_binary(pred_list):
    vec = [0] * len(LABELS)
    for p in pred_list:
        if p in LABELS:
            vec[LABELS.index(p)] = 1
    return vec


# ============================================================
#                     MAIN LOOP
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
        vec = labels_to_binary(labels)

        rows.append([uid] + vec)
        time.sleep(SLEEP_TIME)

    out_df = pd.DataFrame(rows, columns=["id"] + LABELS)
    out_df.to_csv(out_path, index=False)

    print(f"\n✔ Saved → {out_path}\n")


if __name__ == "__main__":
    run()

