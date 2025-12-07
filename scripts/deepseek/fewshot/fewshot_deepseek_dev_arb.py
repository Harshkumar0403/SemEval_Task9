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
LANG = "arb"
TRAIN_FILE = f"Dataset/Processed/{LANG}.tsv"
DEV_FILE = f"Dataset/dev_phase/subtask2/dev/{LANG}.csv"
OUTPUT_DIR = "deepseek/dev/fewshot"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_EXAMPLES = 10
SLEEP_TIME = 0.01  # supports 100 rpm

LABELS = ["Political", "Racial/Ethnic", "Religious", "Gender/Sexual", "Other"]


# ============================================================
#                API KEY
# ============================================================
api_key = os.getenv("DEEPSEEK_API_KEY")
if api_key is None:
    raise ValueError("DEEPSEEK_API_KEY not set in environment!")

client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")


# ============================================================
#  Load few-shot examples from processed training (ARABIC)
# ============================================================
def load_fewshot_examples():
    df = pd.read_csv(TRAIN_FILE, sep="\t")

    df = df[df["labels"] != "None"]  # keep meaningful examples

    examples = df.sample(min(MAX_EXAMPLES, len(df)), random_state=42)

    txt = ""
    for _, row in examples.iterrows():
        t = row["text"]
        l = row["labels"]
        txt += f"Text: {t}\nLabels: {l}\n\n"

    return txt.strip()


fewshot_text = load_fewshot_examples()


# ============================================================
#                SYSTEM PROMPT — FEWSHOT (ARABIC)
# ============================================================
SYSTEM_PROMPT = f"""
You are an expert analyst specializing in detecting hateful, abusive, or targeted
content in Arabic social-media posts. You must classify each post into ANY number
of the categories below (multi-label possible):

1. Political
2. Racial/Ethnic
3. Religious
4. Gender/Sexual
5. Other
6. None — ONLY if the text contains no harmful, abusive, or targeted content.

Rules:
- Output MUST be a JSON list, e.g.: ["Political", "Racial/Ethnic"].
- Do NOT add explanations.
- Do NOT generate prose.
- If no category applies, return ["None"].

Guidance:
Arabic posts often contain political criticism, sectarian references, tribal
identity, gendered slurs, or hostility toward national/ethnic groups. Label strictly
based on explicit hostility or negative group targeting — not implied meaning.

Few-shot examples:

{fewshot_text}

Now classify the following text. Return ONLY:
{{"labels": "<comma-separated list or None>"}}
"""


# ============================================================
#  Utility — Extract labels robustly from GPT output
# ============================================================
def extract_labels(text):
    # Attempt direct JSON
    try:
        obj = json.loads(text)
        if "labels" in obj:
            raw = obj["labels"].strip()
            if raw.lower() == "none":
                return []
            return [x.strip() for x in raw.split(",") if x.strip()]
    except:
        pass

    # Regex fallback for {"labels":"..."}
    match = re.search(r'{"labels":\s*"(.*?)"}', text)
    if match:
        raw = match.group(1)
        if raw.lower() == "none":
            return []
        return [x.strip() for x in raw.split(",")]

    return []


# ============================================================
#               GPT FEWSHOT PREDICTOR
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
#    Convert label list → 5-dimensional binary vector
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
    print(f"\n=== Deepseek FEWSHOT inference for {LANG} ===\n")

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

