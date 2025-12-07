import os
import time
import json
import re
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ============================================================
# API Key
# ============================================================
api_key = os.getenv("DEEPSEEK_API_KEY")
if api_key is None:
    raise ValueError("Error: DEEPSEEK_API_KEY missing. Please export it first.")

client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")

# ============================================================
# SYSTEM PROMPT for Chinese
# ============================================================
SYSTEM_PROMPT = """
You are an expert social-media moderation analyst specializing in hateful, abusive,
or targeted content within Chinese posts. Many Chinese posts in this dataset are 
very short (sometimes only 1–3 characters) but can still contain explicit hostility.

Your task: classify the text into ANY number of these labels (multi-label allowed):

1. Political
2. Racial/Ethnic
3. Religious
4. Gender/Sexual
5. Other
6. None  — ONLY if no harmful or targeted content exists.

Rules:
- OUTPUT MUST BE: a JSON list, e.g. ["Political", "Religious"].
- If nothing applies, output ["None"].
- NO explanations.
- NO prose.
- Classify ONLY explicit group-directed hostility — do NOT guess implied meaning.
- Emojis, numbers, or symbols alone do NOT count as hostility.

Short Chinese posts may rely on slang, abbreviated insults, ethnic markers, political
references, etc. Assign labels only when hostility clearly targets a group.
"""

LABELS = ["Political", "Racial/Ethnic", "Religious", "Gender/Sexual", "Other"]


# ============================================================
# JSON extraction
# ============================================================
def extract_json_list(text):
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except:
        pass

    match = re.search(r"\[.*?\]", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass

    return ["None"]


# ============================================================
# GPT PREDICT
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
                max_tokens=32,
            )

            content = response.choices[0].message.content
            pred_list = extract_json_list(content)

            if len(pred_list) == 1 and pred_list[0].lower() == "none":
                return []

            return pred_list

        except Exception as e:
            print(f"Retry due to error: {e}")
            time.sleep(1)

    return []


# ============================================================
# Convert label list → 5-dim binary vector
# ============================================================
def labels_to_binary(pred_list):
    vec = [0] * 5
    for label in pred_list:
        if label in LABELS:
            vec[LABELS.index(label)] = 1
    return vec


# ============================================================
# MAIN inference for Chinese
# ============================================================
def run_inference():
    lang = "zho"
    print(f"\n=== GPT-4.1-mini ZEROSHOT inference for {lang} ===\n")

    df = pd.read_csv(f"Dataset/dev_phase/subtask2/dev/{lang}.csv")

    os.makedirs("deepseek/dev/zeroshot", exist_ok=True)
    out_path = f"deepseek/dev/zeroshot/pred_{lang}.csv"

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        uid = row["id"]
        text = str(row["text"])

        pred_list = gpt_predict(text)
        vec = labels_to_binary(pred_list)

        results.append([uid] + vec)

        time.sleep(0.01)

    out_df = pd.DataFrame(results, columns=["id"] + LABELS)
    out_df.to_csv(out_path, index=False)

    print(f"\n✓ Saved predictions → {out_path}\n")


if __name__ == "__main__":
    run_inference()

