import os
import time
import json
import requests
import pandas as pd

# ============================================================
#                     CONFIG + PATHS
# ============================================================

DEV_FILE = "Dataset/dev_phase/subtask2/dev/hin.csv"

OUT_DIR = "llama/dev/zeroshot"
os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_FILE = f"{OUT_DIR}/pred_hin.csv"

API_KEY = os.environ.get("NVIDIA_API_KEY")
if API_KEY is None:
    raise ValueError("ERROR: NVIDIA_API_KEY not found in environment!")

INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


# ============================================================
#           HINDI-SPECIFIC ZERO-SHOT SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an expert Hindi social-media polarization analyst. Your task is to classify
Hindi posts into ANY number of the following categories (multi-label allowed):

1. Political
2. Racial/Ethnic
3. Religious
4. Gender/Sexual
5. Other
6. None — ONLY if the text contains no harmful, hateful, or polarized content.

Output rules:
- Output MUST be a JSON list like ["Political", "Religious"].
- DO NOT add explanations, translations, reasoning, or extra text.
- If no category applies, return ["None"].

Guidance for Hindi:
- Hindi posts often contain sarcasm, slurs, code-mixed English, emojis, slang, hashtags,
  and culturally loaded expressions.
- Political: references to सरकार, मोदी, बीजेपी, कांग्रेस, चुनाव, विरोध, देश बचाओ,
  राष्ट्रवाद, protests, policies, political leaders.
- Racial/Ethnic: caste references (जाति), क्षेत्रीय identities, जातिवादी insults,
  community-based targeting.
- Religious: हिंदू/मुस्लिम, मंदिर/मस्जिद, धर्म, धार्मिक slurs, communal tone, sectarian remarks.
- Gender/Sexual: sexism, trolling women, gender insults, misogyny, LGBTQ+ hostility.
- Other: harmful content that does not fit above categories.

Be culturally precise. Do NOT over-classify.
Always respond ONLY with a valid JSON list.
"""


def build_user_prompt(text):
    return f"""
Text: \"{text}\"

Classify this Hindi text into the correct labels. Respond ONLY with a JSON list.
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
        "top_p": 1.0,
        "max_tokens": 128,
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
#              LABEL MAPPING HELPERS
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
    print("\n=== LLAMA DEV SET INFERENCE (HINDI) ===")

    df = pd.read_csv(DEV_FILE)

    # Drop empty gold label columns in dev set
    drop_cols = ["political", "racial/ethnic", "religious", "gender/sexual", "other"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    preds = []

    for i, row in df.iterrows():
        text_id = row["id"]
        text = row["text"]

        print(f"[{i+1}/{len(df)}] ID={text_id}")

        user_prompt = build_user_prompt(text)
        raw_pred = call_llama(user_prompt)

        # Parse JSON output safely
        try:
            parsed = json.loads(raw_pred)
            if not isinstance(parsed, list):
                parsed = ["None"]
        except:
            parsed = ["None"]

        binary = convert_json_to_binary(parsed)
        preds.append([text_id] + binary)

        time.sleep(0.15)

    # Save predictions
    out_df = pd.DataFrame(preds, columns=["id"] + LABELS)
    out_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\nPredictions saved → {OUTPUT_FILE}")
    print("=== DONE ===\n")


if __name__ == "__main__":
    main()

