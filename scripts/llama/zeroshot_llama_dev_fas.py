import os
import time
import json
import requests
import pandas as pd

# ============================================================
#                     CONFIG + PATHS
# ============================================================

DEV_FILE = "Dataset/dev_phase/subtask2/dev/fas.csv"

OUT_DIR = "llama/dev/zeroshot"
os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_FILE = f"{OUT_DIR}/pred_fas.csv"

API_KEY = os.environ.get("NVIDIA_API_KEY")
if API_KEY is None:
    raise ValueError("ERROR: NVIDIA_API_KEY not found in environment!")

INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


# ============================================================
#          PERSIAN-SPECIFIC ZERO-SHOT SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an expert Persian (Farsi) social-media polarization analyst. Your task is to
classify Persian posts into ANY number of the following categories (multi-label allowed):

1. Political
2. Racial/Ethnic
3. Religious
4. Gender/Sexual
5. Other
6. None — ONLY if the post contains no harmful, abusive, hostile, targeted, or
   polarization-inducing content.

Output Rules:
- Output MUST be a JSON list, e.g., ["Political", "Other"].
- DO NOT provide explanations, analysis, or translation ─ only the JSON list.
- If nothing applies, return ["None"].

Guidance for Persian content:
- Political: حکومت, دولت, جمهوری اسلامی, انتخابات, اعتراضات, آزادی بیان, سیاستمداران,
  نظام, corruption, sanctions, opposition.
- Racial/Ethnic: قومیت‌گرایی, references to Afghans, Arabs, Kurds, Turks, Baluch, or
  discriminatory/derogatory ethnic remarks.
- Religious: اسلام, شیعه, سنی, روحانیت, حوزه, قرآن, توهین مذهبی, faith-based hostility.
- Gender/Sexual: زنان, حجاب, آزادی زنان, تبعیض جنسیتی, LGBTQ+ hostility, sexual insults.
- Other: harmful, degrading, or violent content not belonging to the above classes.

Persian social media frequently uses sarcasm, satire, political frustration, coded criticism,
and cultural-religious references. Classify conservatively but accurately.

Always respond ONLY with the final JSON list.
"""


def build_user_prompt(text):
    return f"""
Text: \"{text}\"

Classify this Persian text into the correct labels.
Respond ONLY with a JSON list.
"""


# ============================================================
#                     API CALL FUNCTION
# ============================================================

def call_llama(prompt, retries=5, sleep=0.2):
    """NVIDIA LLAMA API call with retry + incremental backoff."""
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
    print("\n=== LLAMA DEV SET INFERENCE (PERSIAN / FAS) ===")

    df = pd.read_csv(DEV_FILE)

    # Drop empty gold label columns (dev file)
    drop_cols = ["political", "racial/ethnic", "religious", "gender/sexual", "other"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    preds = []

    for i, row in df.iterrows():
        text_id = row["id"]
        text = row["text"]

        print(f"[{i+1}/{len(df)}] ID={text_id}")

        user_prompt = build_user_prompt(text)
        raw_pred = call_llama(user_prompt)

        # safe JSON parsing
        try:
            parsed = json.loads(raw_pred)
            if not isinstance(parsed, list):
                parsed = ["None"]
        except:
            parsed = ["None"]

        binary = convert_json_to_binary(parsed)
        preds.append([text_id] + binary)

        time.sleep(0.15)  # rate limit safety

    # Save results
    out_df = pd.DataFrame(preds, columns=["id"] + LABELS)
    out_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\nPredictions saved → {OUTPUT_FILE}")
    print("=== DONE ===\n")


if __name__ == "__main__":
    main()

