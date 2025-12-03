import os
import time
import json
import requests
import pandas as pd

# ============================================================
#               CONFIG + DIRECTORY SETUP
# ============================================================

DEV_FILE = "Dataset/dev_phase/subtask2/dev/amh.csv"
OUT_DIR = "llama/dev/zeroshot"
os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_FILE = f"{OUT_DIR}/pred_amh.csv"

API_KEY = os.environ.get("NVIDIA_API_KEY")
if API_KEY is None:
    raise ValueError("ERROR: NVIDIA_API_KEY not found in environment!")

INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


# ============================================================
#       AMHARIC - SPECIFIC SYSTEM PROMPT (ZERO-SHOT)
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


def build_user_prompt(text):
    return f"""
Text: \"{text}\"

Classify the text into the correct labels. Respond ONLY with a JSON list.
"""


# ============================================================
#                    LLAMA INFERENCE
# ============================================================

def call_llama(prompt, retries=5, sleep=0.2):
    """NVIDIA NIM LLAMA API call with retry strategy."""
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
            response = requests.post(INVOKE_URL, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]

            print(f"[WARN] Status {response.status_code}: {response.text}")
            time.sleep(sleep * (attempt + 1))

        except Exception as e:
            print(f"[ERROR] {e} — retrying...")
            time.sleep(sleep * (attempt + 1))

    return "[]"


# ============================================================
#               LABEL PROCESSING HELPERS
# ============================================================

LABELS = ["Political", "Racial/Ethnic", "Religious", "Gender/Sexual", "Other"]

def convert_json_to_binary(label_list):
    if "None" in label_list:
        return [0, 0, 0, 0, 0]

    return [1 if lbl in label_list else 0 for lbl in LABELS]


# ============================================================
#                     MAIN INFERENCE
# ============================================================

def main():
    print("\n=== LLAMA DEV SET INFERENCE (AMHARIC) ===")

    df = pd.read_csv(DEV_FILE)

    # remove empty gold columns if present
    drop_cols = ["political", "racial/ethnic", "religious", "gender/sexual", "other"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    preds = []

    for i, row in df.iterrows():
        text_id = row["id"]
        text = row["text"]

        print(f"[{i+1}/{len(df)}] ID={text_id}")

        user_prompt = build_user_prompt(text)
        raw_pred = call_llama(user_prompt)

        # Parse LLAMA JSON output safely
        try:
            parsed = json.loads(raw_pred)
            if not isinstance(parsed, list):
                parsed = ["None"]
        except:
            parsed = ["None"]

        binary = convert_json_to_binary(parsed)
        preds.append([text_id] + binary)

        time.sleep(0.15)   # rate-limit safety

    # save predictions
    out_df = pd.DataFrame(preds, columns=["id"] + LABELS)
    out_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\nPredictions saved → {OUTPUT_FILE}")
    print("=== DONE ===\n")


if __name__ == "__main__":
    main()

