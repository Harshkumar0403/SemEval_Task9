import os
import time
import json
import re
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ============================================================
# Load API Key
# ============================================================
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("Error: OPENAI_API_KEY not found. Please export it first.")

client = OpenAI(api_key=api_key)

# ============================================================
# SYSTEM PROMPT (Nepali — expert moderation classifier)
# ============================================================
SYSTEM_PROMPT = """
You are an expert analyst specializing in detecting hateful, abusive, or targeted
content in Nepali social-media posts. You must classify each post into ANY number
of the categories below (multi-label allowed):

1. Political
2. Racial/Ethnic
3. Religious
4. Gender/Sexual
5. Other
6. None — ONLY if the text contains no harmful, abusive, or targeted content.

Rules:
- Output MUST be a JSON list, for example: ["Political", "Religious"].
- No explanations. No descriptions. Only the JSON list.
- If no category applies, return ["None"].

Nepali posts often include political conflict, caste/ethnic hostility, regional tensions,
religious insults, gender/sexual harassment, or coded slurs. Be strict when hostility is
directed toward any identifiable social group.
"""

LABELS = ["Political", "Racial/Ethnic", "Religious", "Gender/Sexual", "Other"]

# ============================================================
# Extract JSON array from model output
# ============================================================
def extract_json_list(text):
    # Direct JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except:
        pass

    # Regex fallback
    match = re.search(r"\[.*?\]", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass

    return ["None"]

# ============================================================
# GPT Call
# ============================================================
def gpt_predict(text):
    for _ in range(5):  # retry up to 5 times
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                temperature=0.0,
                max_tokens=64,
            )

            content = response.choices[0].message.content
            pred_list = extract_json_list(content)

            # Convert ["None"] → empty list
            if len(pred_list) == 1 and pred_list[0].lower() == "none":
                return []

            return pred_list

        except Exception as e:
            print(f"Retry due to error: {e}")
            time.sleep(1)

    return []  # fallback if all retries fail

# ============================================================
# Convert labels → binary 5-d vector
# ============================================================
def labels_to_binary(pred_list):
    vec = [0] * 5
    for label in pred_list:
        if label in LABELS:
            vec[LABELS.index(label)] = 1
    return vec

# ============================================================
# Main Inference
# ============================================================
def run_inference():
    lang = "nep"
    print(f"\n=== GPT-4.1-mini ZEROSHOT inference for {lang} ===\n")

    df = pd.read_csv(f"Dataset/dev_phase/subtask2/dev/{lang}.csv")

    os.makedirs("gpt/dev/zeroshot", exist_ok=True)
    out_path = f"gpt/dev/zeroshot/pred_{lang}.csv"

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        uid = row["id"]
        text = str(row["text"])

        pred = gpt_predict(text)
        vec = labels_to_binary(pred)

        rows.append([uid] + vec)

        time.sleep(0.01)  # 100 RPM safety

    out_df = pd.DataFrame(rows, columns=["id"] + LABELS)
    out_df.to_csv(out_path, index=False)

    print(f"\nSaved predictions → {out_path}\n")

if __name__ == "__main__":
    run_inference()

