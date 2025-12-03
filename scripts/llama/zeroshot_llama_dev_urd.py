import os
import time
import json
import requests
import pandas as pd

# ============================================================
#                   CONFIGURATION + PATHS
# ============================================================

DEV_FILE = "Dataset/dev_phase/subtask2/dev/urd.csv"

OUT_DIR = "llama/dev/zeroshot"
os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_FILE = f"{OUT_DIR}/pred_urd.csv"

API_KEY = os.environ.get("NVIDIA_API_KEY")
if API_KEY is None:
    raise ValueError("ERROR: NVIDIA_API_KEY is not set in environment!")

INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


# ============================================================
#        URDU-SPECIFIC ZERO-SHOT SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an expert Urdu social-media analyst specializing in detecting harmful,
polarizing, or discriminatory content. Your task is MULTI-LABEL classification using
the following categories:

1. Political
2. Racial/Ethnic
3. Religious
4. Gender/Sexual
5. Other
6. None — ONLY if the text does not contain harmful, abusive, hateful, or
   polarization-related content.

Output Rules:
- Respond ONLY with a JSON list, e.g., ["Political", "Religious"].
- No explanation, no translation, no justification.
- If nothing applies, return ["None"].

Guidance for Urdu:
- Political: حکومت، سیاست، انتخاب، پارلیمنٹ، سیاسی جماعتیں، ریاستی پالیسیاں۔
- Religious: اسلام، فرقہ واریت، مذہبی تنقید، مقدسات کے بارے میں منفی بات۔
- Racial/Ethnic: پاکستانی، ہندوستانی، مہاجر، افغان، پشتون، پنجابی، نسلی تعصب۔
- Gender/Sexual: عورت مخالف گفتگو، ہراسانی، LGBTQ+ مخالف بات۔
- Other: اگر مواد جارحانہ/گالی/توہین آمیز ہو لیکن اوپر کی 4 اقسام میں نہ آئے۔

Return ONLY the JSON list.
"""


def build_user_prompt(text):
    return f"""
Text: "{text}"

Classify this Urdu text into all relevant labels.
Return ONLY a JSON list.
"""


# ============================================================
#                     API CALL FUNCTION
# ============================================================

def call_llama(prompt, retries=5, sleep=0.2):
    """Call NVIDIA LLAMA API with retry + incremental backoff."""
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
#                     LABEL PROCESSING
# ============================================================

LABELS = ["Political", "Racial/Ethnic", "Religious", "Gender/Sexual", "Other"]

def convert_json_to_binary(label_list):
    """Convert model's label list → five-column binary."""
    if "None" in label_list:
        return [0, 0, 0, 0, 0]
    return [1 if lbl in label_list else 0 for lbl in LABELS]


# ============================================================
#                           MAIN
# ============================================================

def main():
    print("\n=== LLAMA DEV SET INFERENCE (URDU / URD) ===")

    df = pd.read_csv(DEV_FILE)

    # Remove empty gold-column placeholders
    drop_cols = ["political", "racial/ethnic", "religious", "gender/sexual", "other"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    preds = []

    for i, row in df.iterrows():
        text_id = row["id"]
        text = row["text"]

        print(f"[{i+1}/{len(df)}] ID={text_id}")

        prompt = build_user_prompt(text)
        raw_pred = call_llama(prompt)

        # Parse JSON safely
        try:
            parsed = json.loads(raw_pred)
            if not isinstance(parsed, list):
                parsed = ["None"]
        except:
            parsed = ["None"]

        binary = convert_json_to_binary(parsed)
        preds.append([text_id] + binary)

        time.sleep(0.15)  # small delay for rate limits

    # Save predictions
    out_df = pd.DataFrame(preds, columns=["id"] + LABELS)
    out_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\nPredictions saved → {OUTPUT_FILE}")
    print("=== DONE ===\n")


if __name__ == "__main__":
    main()

