#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import os
import unicodedata
import regex as re
from collections import Counter

# -------------------------------------------------------------
# Helper: Normalize text using Unicode NFC
# -------------------------------------------------------------
def normalize_text(text):
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFC", text)


# -------------------------------------------------------------
# Helper: Extract emojis using regex library
# -------------------------------------------------------------
def extract_emojis(text):
    emoji_pattern = re.compile(r"\p{Emoji}", flags=re.UNICODE)
    return emoji_pattern.findall(text)


# -------------------------------------------------------------
# Convert numeric labels → text labels
# -------------------------------------------------------------
def convert_labels(row):
    label_map = {
        "political": "Political",
        "racial/ethnic": "Racial/Ethnic",
        "religious": "Religious",
        "gender/sexual": "Gender/Sexual",
        "other": "Other"
    }

    labels = []

    for key, label_text in label_map.items():
        if row[key] == 1:
            labels.append(label_text)

    if len(labels) == 0:
        return "None"
    return ", ".join(labels)


# -------------------------------------------------------------
# Perform analysis and save to text file
# -------------------------------------------------------------
def save_analysis(df, lang, output_path):
    total_samples = len(df)

    # Count each label occurrence
    label_counter = Counter()

    multi_label_count = 0
    single_label_count = 0
    none_count = 0

    emoji_counter = Counter()
    text_lengths = []

    for _, row in df.iterrows():
        labels = row["labels"]
        text = row["text"]

        # Text length
        text_lengths.append(len(text.split()))

        # Label statistics
        if labels == "None":
            none_count += 1
        else:
            split_labels = [l.strip() for l in labels.split(",")]
            if len(split_labels) > 1:
                multi_label_count += 1
            else:
                single_label_count += 1

            for lbl in split_labels:
                label_counter[lbl] += 1

        # Emoji stats
        for e in extract_emojis(text):
            emoji_counter[e] += 1

    dominant_class = label_counter.most_common(1)[0][0] if label_counter else "None"

    avg_len = sum(text_lengths) / len(text_lengths)
    max_len = max(text_lengths)
    min_len = min(text_lengths)

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Language: {lang}\n")
        f.write(f"Total Samples: {total_samples}\n\n")

        f.write("Label Distribution:\n")
        for lbl, count in label_counter.items():
            pct = (count / total_samples) * 100
            f.write(f"  {lbl}: {count} ({pct:.2f}%)\n")

        f.write(f"\nDominant Class: {dominant_class}\n\n")

        f.write(f"Zero-label instances: {none_count}\n")
        f.write(f"Single-label instances: {single_label_count}\n")
        f.write(f"Multi-label instances: {multi_label_count}\n\n")

        f.write("Text Length Stats (word count):\n")
        f.write(f"  Average length: {avg_len:.2f}\n")
        f.write(f"  Min length: {min_len}\n")
        f.write(f"  Max length: {max_len}\n\n")

        f.write("Emoji Statistics:\n")
        total_emojis = sum(emoji_counter.values())
        f.write(f"  Total Emojis: {total_emojis}\n")

        top_emojis = emoji_counter.most_common(10)
        f.write("  Top Emojis:\n")
        for e, c in top_emojis:
            f.write(f"    {e} : {c}\n")

        f.write("\nAnalysis Complete.\n")


# -------------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Preprocess SemEval Task 9 Multilingual Data")
    parser.add_argument("--lang", type=str, required=True, help="Language code, e.g. --lang eng")
    args = parser.parse_args()

    lang = args.lang

    # Input and output paths
    input_path = f"Dataset/dev_phase/subtask2/train/{lang}.csv"
    processed_dir = "Dataset/Processed"
    analysis_dir = "Dataset/Analysis"

    # Create dirs if missing
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)

    processed_output = f"{processed_dir}/{lang}.tsv"
    analysis_output = f"{analysis_dir}/{lang}.txt"

    # Load dataset
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)

    # Normalize text
    df["text"] = df["text"].astype(str).apply(normalize_text)

    # Convert numeric labels to textual labels
    df["labels"] = df.apply(convert_labels, axis=1)

    # Save processed file
    df_out = df[["id", "text", "labels"]]
    df_out.to_csv(processed_output, sep="\t", index=False, encoding="utf-8")
    print(f"Processed file saved to: {processed_output}")

    # Run analysis
    save_analysis(df_out, lang, analysis_output)
    print(f"Analysis file saved to: {analysis_output}")


if __name__ == "__main__":
    main()

