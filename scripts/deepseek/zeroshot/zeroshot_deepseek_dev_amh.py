import os
import time
import json
import re
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ============================================================
# Load API key
# ============================================================
api_key = os.getenv("DEEPSEEK_API_KEY")
if api_key is None:
    raise ValueError("Error: DEEPSEEK_API_KEY not found. Please export it first.")

client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")

# ============================================================
# SYSTEM PROMPT (Amharic — your exact text)
# ============================================================
SYSTEM_PROMPT = """
You are an expert analyst specializing in detecting hateful, abusive, or targeted
content in Amharic social-media posts. You must classify each post into ANY number
of the categories below (multi-label possible):

1. Political
2. Racial/Ethnic
3. Religious
4. Gender/Sexual
5. Other
6. None  — ONLY if the text contains no harmful, abusive, or targeted content.

Rules:
- Output MUST be a JSON list, for example: ["Political", "Racial/Ethnic"].
- Do NOT add explanations.
- Do NOT generate prose.
- If no category applies, return ["None"].

Be strict when evaluating political or ethnic commentary. Amharic content often mentions
political parties, government actors, ethnic groups, conflicts, or regional tensions—
classify these as Political or Racial/Ethnic when relevant.
"""

LABELS = ["Political", "Racial/Ethnic", "Religious", "Gender/Sexual", "Other"]

# ============================================================
# Robust JSON extraction helper
# ============================================================
def extract_json_list(text):
    """
    Extracts a JSON list like ["Political", "Religious"] even if formatting is broken.
    """

    # Try direct JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except:
        pass

    # Try regex extraction of first bracketed list
    match = re.search(r"\[.*?\]", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass

    return ["None"]


# ============================================================
# Deepseek prediction (robust)
# ============================================================
def gpt_predict(text):
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                max_tokens=64,
                temperature=0.0,
            )

            content = response.choices[0].message.content
            pred_list = extract_json_list(content)

            # Normalize NONE
            if len(pred_list) == 1 and pred_list[0].lower() == "none":
                return []

            return pred_list

        except Exception as e:
            print(f"Retry due to error: {e}")
            time.sleep(1)

    return []


# ============================================================
# Convert predicted labels → 5-vector
# ============================================================
def labels_to_binary(pred_list):
    vec = [0] * 5
    for label in pred_list:
        if label in LABELS:
            vec[LABELS.index(label)] = 1
    return vec


# ============================================================
# Main inference pipeline
# ============================================================
def run_inference():
    lang = "amh"
    print(f"\n=== Deepseek ZEROSHOT inference for {lang} ===\n")

    df = pd.read_csv(f"Dataset/dev_phase/subtask2/dev/{lang}.csv")

    os.makedirs("deepseek/dev/zeroshot", exist_ok=True)
    output_path = f"deepseek/dev/zeroshot/pred_{lang}.csv"

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        uid = row["id"]
        text = str(row["text"])

        pred = gpt_predict(text)
        vec = labels_to_binary(pred)

        rows.append([uid] + vec)

        time.sleep(0.01)  # respect rate limit

    out_df = pd.DataFrame(rows, columns=["id"] + LABELS)
    out_df.to_csv(output_path, index=False)

    print(f"\nSaved predictions → {output_path}\n")


if __name__ == "__main__":
    run_inference()

