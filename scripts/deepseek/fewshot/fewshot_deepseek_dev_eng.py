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
LANG = "eng"
TRAIN_FILE = f"Dataset/Processed/{LANG}.tsv"
DEV_FILE = f"Dataset/dev_phase/subtask2/dev/{LANG}.csv"
OUTPUT_DIR = "deepseek/dev/fewshot"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_EXAMPLES = 10   # few-shot count
SLEEP_TIME = 0.01   # for 100 RPM limit safety

LABELS = ["Political", "Racial/Ethnic", "Religious", "Gender/Sexual", "Other"]

# ============================================================
#                API KEY
# ============================================================
api_key = os.getenv("DEEPSEEK_API_KEY")
if api_key is None:
    raise ValueError("Error: DEEPSEEK_API_KEY not set!")

client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")


# ============================================================
#     Load few-shot examples from processed training file
# ============================================================
def load_fewshot_examples():
    df = pd.read_csv(TRAIN_FILE, sep="\t")

    df = df[df["labels"] != "None"]

    examples = df.sample(min(MAX_EXAMPLES, len(df)), random_state=42)

    txt = ""
    for _, row in examples.iterrows():
        text = row["text"]
        labels = row["labels"]
        txt += f"Text: {text}\nLabels: {labels}\n\n"

    return txt.strip()


fewshot_text = load_fewshot_examples()


# ============================================================
#               SYSTEM PROMPT — FEWSHOT (ENGLISH)
# ============================================================
SYSTEM_PROMPT = f"""
You are an expert analyst of English social media discourse. You classify short English posts 
for SemEval 2026 Task 9 (Subtask 2): Polarization Type Classification.

English posts often contain political satire, racial hostility, religious attacks, gendered harassment, 
or general group-based hostility. Your task is to detect ONLY explicit polarization directed at 
groups, NOT general negativity.

VALID LABELS:
Political, Racial/Ethnic, Religious, Gender/Sexual, Other, None

DEFINITIONS:
• Political — hostility toward parties, leaders, government, ideologies, laws.
• Racial/Ethnic — hostility toward Black, White, Asian, Latino, immigrants, etc.
• Religious — hostility toward Christians, Muslims, Jews, atheists, etc.
• Gender/Sexual — misogyny, homophobic/transphobic insults, gender-targeted abuse.
• Other — group-based hostility not covered above.
• None — if no explicit group-targeted hostility exists.

IMPORTANT:
• Classify ONLY explicit hostility or negative generalization toward a group.
• Do NOT infer hidden meaning or sarcasm unless explicitly stated.
• A post may have multiple labels.
• Emojis alone do not imply hostility.

Few-shot examples:

{fewshot_text}

Now classify the following text. Return ONLY in this JSON format:
{{"labels": "<comma-separated list or None>"}}
"""


# ============================================================
#  Utility — Extract JSON list even if GPT formats weirdly
# ============================================================
def extract_labels(text):
    try:
        obj = json.loads(text)
        if "labels" in obj:
            raw = obj["labels"].strip()
            if raw.lower() == "none":
                return []
            return [x.strip() for x in raw.split(",") if x.strip()]
    except:
        pass

    match = re.search(r'{"labels":\s*"(.*?)"}', text)
    if match:
        items = match.group(1)
        if items.lower() == "none":
            return []
        return [x.strip() for x in items.split(",")]

    return []


# ============================================================
#                GPT FEWSHOT PREDICTION
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
            labels = extract_labels(content)

            return labels

        except Exception as e:
            print("Retry due to error:", e)
            time.sleep(1)

    return []


# ============================================================
#      Convert label list → binary vector
# ============================================================
def labels_to_binary(pred_list):
    vec = [0] * len(LABELS)
    for label in pred_list:
        if label in LABELS:
            vec[LABELS.index(label)] = 1
    return vec


# ============================================================
#                     MAIN INFERENCE
# ============================================================
def run():
    print(f"\n=== Deepseek FEWSHOT Inference for {LANG} ===\n")

    df = pd.read_csv(DEV_FILE)
    out_path = f"{OUTPUT_DIR}/pred_{LANG}.csv"

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        uid = row["id"]
        text = str(row["text"])

        pred_labels = gpt_predict(text)
        binary = labels_to_binary(pred_labels)

        results.append([uid] + binary)

        time.sleep(SLEEP_TIME)

    out_df = pd.DataFrame(results, columns=["id"] + LABELS)
    out_df.to_csv(out_path, index=False)

    print(f"\n✔ Saved predictions → {out_path}\n")


if __name__ == "__main__":
    run()

