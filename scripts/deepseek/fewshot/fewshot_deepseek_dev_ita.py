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
LANG = "ita"
TRAIN_FILE = f"Dataset/Processed/{LANG}.tsv"
DEV_FILE = f"Dataset/dev_phase/subtask2/dev/{LANG}.csv"
OUTPUT_DIR = "deepseek/dev/fewshot"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_EXAMPLES = 10
SLEEP_TIME = 0.01  # 100 rpm

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
#            SYSTEM PROMPT (ITALIAN FEWSHOT)
# ============================================================
SYSTEM_PROMPT = f"""
You are an expert analyst specializing in detecting hateful, abusive, or targeted
content in Italian social-media posts. You must classify each post into ANY number
of the categories below (multi-label allowed):

1. Political
2. Racial/Ethnic
3. Religious
4. Gender/Sexual
5. Other
6. None — ONLY if the text contains no harmful or targeted content.

Rules:
- Output MUST be a JSON list (e.g., ["Political", "Religious"]).
- NO explanations. NO prose.
- If no category applies, return ["None"].

Guidance:
Italian social-media content often includes political hostility (partiti politici,
governo, ideologie), racial/ethnic hostility (immigrati, minoranze, gruppi etnici),
religious attacks (cristiani, musulmani, ebrei), and gender/sexual harassment
(misoginia, attacchi LGBTQ). Label ONLY explicit hostility or group-targeting.

Few-shot examples:

{fewshot_text}

Now classify the following text. Return ONLY:
{{"labels": "<comma-separated list or None>"}}
"""


# ============================================================
#       JSON LABEL EXTRACTION (robust against noise)
# ============================================================
def extract_labels(text):
    try:
        obj = json.loads(text)
        if "labels" in obj:
            raw = obj["labels"]
            if raw.lower() == "none":
                return []
            return [x.strip() for x in raw.split(",") if x.strip()]
    except:
        pass

    match = re.search(r'{"labels":\s*"(.*?)"}', text)
    if match:
        raw = match.group(1)
        if raw.lower() == "none":
            return []
        return [x.strip() for x in raw.split(",")]

    return []


# ============================================================
#                   GPT CALL (with retries)
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
#     Convert list of predicted labels → binary vector
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

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        uid = row["id"]
        text = str(row["text"])

        pred_labels = gpt_predict(text)
        binary_vec = labels_to_binary(pred_labels)

        results.append([uid] + binary_vec)
        time.sleep(SLEEP_TIME)

    out_df = pd.DataFrame(results, columns=["id"] + LABELS)
    out_df.to_csv(out_path, index=False)

    print(f"\n✔ Saved predictions → {out_path}\n")


if __name__ == "__main__":
    run()

