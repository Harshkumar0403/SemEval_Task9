import os
import time
import json
import random
import requests
import pandas as pd

# ============================================================
#               CONFIG + DIRECTORY SETUP
# ============================================================

DEV_FILE = "Dataset/dev_phase/subtask2/dev/eng.csv"
OUT_DIR = "llama/dev/zeroshot"
os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_FILE = f"{OUT_DIR}/pred_eng.csv"

API_KEY = os.environ.get("NVIDIA_API_KEY")
if API_KEY is None:
    raise ValueError("ERROR: NVIDIA_API_KEY not found in environment!")

INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# ============================================================
#                 SYSTEM + USER PROMPT TEMPLATE
# ============================================================

SYSTEM_PROMPT = """
You are an expert social-media moderation analyst specializing in hateful, abusive,
or targeted content classification. You must classify each input text into ANY number 
of the following labels (multi-label possible):

1. Political
2. Racial/Ethnic
3. Religious
4. Gender/Sexual
5. Other
6. None  (ONLY if absolutely no harmful/targeted content exists)

Rules:
- Return ONLY a JSON list of labels, e.g. ["Political", "Religious"].
- If no category applies, return ["None"].
- Do NOT explain anything.
- Do NOT output prose.
"""

def build_user_prompt(text):
    return f"""
Text: \"{text}\"

Classify the text into the correct labels. Respond ONLY with a JSON list.
"""

# ============================================================
#                    LLAMA INFERENCE
# ============================================================

def call_llama(prompt, retries=5, sleep=0.2):
    """NVIDIA NIM call with retry handling."""
    payload = {
        "model": "meta/llama-4-maverick-17b-128e-instruct",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
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
            resp = requests.post(INVOKE_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                out = resp.json()
                return out["choices"][0]["message"]["content"]
            else:
                print(f"[WARN] Status {resp.status_code}: {resp.text}")
                time.sleep(sleep * (attempt + 1))
        except Exception as e:
            print(f"[ERROR] {e}, retrying...")
            time.sleep(sleep * (attempt + 1))

    return "[]"

# ============================================================
#                    LABEL PROCESSING
# ============================================================

LABELS = ["Political", "Racial/Ethnic", "Religious", "Gender/Sexual", "Other"]

def convert_json_to_binary(label_list):
    """Convert predicted label list into 5-column binary vector."""
    if "None" in label_list:
        return [0, 0, 0, 0, 0]

    out = []
    for lbl in LABELS:
        out.append(1 if lbl in label_list else 0)
    return out

# ============================================================
#                      MAIN INFERENCE
# ============================================================

def main():
    print("\n=== LLAMA DEV SET INFERENCE (ENGLISH) ===")

    df = pd.read_csv(DEV_FILE)

    # Remove empty label columns if present
    columns_to_drop = ["political", "racial/ethnic", "religious", "gender/sexual", "other"]
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])

    preds = []

    for idx, row in df.iterrows():
        text_id = row["id"]
        text = row["text"]

        print(f"[{idx+1}/{len(df)}] ID={text_id}")

        prompt = build_user_prompt(text)
        raw_pred = call_llama(prompt)

        try:
            parsed = json.loads(raw_pred)
            if not isinstance(parsed, list):
                parsed = ["None"]
        except:
            parsed = ["None"]

        binary = convert_json_to_binary(parsed)
        preds.append([text_id] + binary)

        time.sleep(0.15)  # rate limit safety

    # SAVE OUTPUT CSV
    out_df = pd.DataFrame(preds, columns=["id"] + LABELS)
    out_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\nPredictions saved → {OUTPUT_FILE}")
    print("=== DONE ===\n")


if __name__ == "__main__":
    main()

