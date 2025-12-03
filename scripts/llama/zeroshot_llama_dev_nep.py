import os
import time
import json
import requests
import pandas as pd

# ============================================================
#                     CONFIG + PATHS
# ============================================================

DEV_FILE = "Dataset/dev_phase/subtask2/dev/nep.csv"

OUT_DIR = "llama/dev/zeroshot"
os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_FILE = f"{OUT_DIR}/pred_nep.csv"

API_KEY = os.environ.get("NVIDIA_API_KEY")
if API_KEY is None:
    raise ValueError("ERROR: NVIDIA_API_KEY not found in environment!")

INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


# ============================================================
#          NEPALI-SPECIFIC ZERO-SHOT SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an expert Nepali social-media content analyst. Your task is to classify
Nepali posts into ANY number of the following categories (multi-label allowed):

1. Political
2. Racial/Ethnic
3. Religious
4. Gender/Sexual
5. Other
6. None — ONLY if the text contains no harmful, abusive, divisive, or polarized content.

Output rules:
- Output MUST be a JSON list, e.g., ["Political", "Religious"].
- NO explanations, NO translation, NO commentary — only the JSON list.
- If nothing applies, output ["None"].

Guidance for Nepali:
- Common features: political slogans, party/leader references, caste/ethnicity mentions,
  religious identity comments, sarcastic remarks, short informal text, and Nepali-English code-mixing.
- Political: सरकार, पार्टी, एमाले, कांग्रेस, चुनाव, नेताहरू, संविधान, विरोध.
- Racial/Ethnic: जात, जातीय समूह, क्षेत्रीय पहिचान, भेदभावपूर्ण टिप्पणी.
- Religious: हिन्दू, मुस्लिम, धार्मिक भावनाहरू, धर्मसँग सम्बन्धित अपमान.
- Gender/Sexual: लैंगिक पक्षपात, महिलाविरुद्ध टिप्पणी, LGBTQ+ hostility.
- Other: harmful or hostile content not fitting above categories.

Be strict but balanced. Do NOT assign labels unless clearly present.

Always respond ONLY with a valid JSON list.
"""


def build_user_prompt(text):
    return f"""
Text: \"{text}\"

Classify this Nepali text into the correct labels.
Respond ONLY with a JSON list.
"""


# ============================================================
#                     API CALL FUNCTION
# ============================================================

def call_llama(prompt, retries=5, sleep=0.2):
    """NVIDIA LLAMA API call with retry logic."""
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

            print(f"[WARN] Status {r.status_code}: {r.text}")
            time.sleep(sleep * (attempt + 1))

        except Exception as e:
            print(f"[ERROR] {e} — retrying...")
            time.sleep(sleep * (attempt + 1))

    return "[]"


# ============================================================
#              LABEL PROCESSING HELPERS
# ============================================================

LABELS = ["Political", "Racial/Ethnic", "Religious", "Gender/Sexual", "Other"]

def convert_json_to_binary(label_list):
    if "None" in label_list:
        return [0, 0, 0, 0, 0]
    return [1 if lbl in label_list else 0 for lbl in LABELS]


# ============================================================
#                         MAIN
# ============================================================

def main():
    print("\n=== LLAMA DEV SET INFERENCE (NEPALI) ===")

    df = pd.read_csv(DEV_FILE)

    # Remove empty gold label columns from dev set
    drop_cols = ["political", "racial/ethnic", "religious", "gender/sexual", "other"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    preds = []

    for i, row in df.iterrows():
        text_id = row["id"]
        text = row["text"]

        print(f"[{i+1}/{len(df)}] ID={text_id}")

        user_prompt = build_user_prompt(text)
        raw_pred = call_llama(user_prompt)

        # Safe JSON parsing
        try:
            parsed = json.loads(raw_pred)
            if not isinstance(parsed, list):
                parsed = ["None"]
        except:
            parsed = ["None"]

        binary = convert_json_to_binary(parsed)
        preds.append([text_id] + binary)

        time.sleep(0.15)  # avoid rate limit issues

    # Save final predictions
    out_df = pd.DataFrame(preds, columns=["id"] + LABELS)
    out_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\nPredictions saved → {OUTPUT_FILE}")
    print("=== DONE ===\n")


if __name__ == "__main__":
    main()

