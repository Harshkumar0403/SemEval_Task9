#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Languages to process
LANGUAGES = ["amh", "arb", "deu", "eng", "fas", "hau", "hin",
             "ita", "nep", "spa", "tur", "urd", "zho"]

INPUT_DIR = "Dataset/Processed"
OUTPUT_DIR = "Dataset/Subset"

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

def stratified_sample(df, n=200):
    """
    Stratified sampling by multi-label 'labels' column.
    If a class has very few samples, fall back to simple sampling.
    """

    # If unique label groups < 5 or rare classes exist → safer to sample normally
    label_counts = df["labels"].value_counts()

    # If sampling fails, fallback to simple random selection
    try:
        # Use labels column as stratification target
        df_sample, _ = train_test_split(
            df,
            train_size=min(n, len(df)),
            stratify=df["labels"],
            random_state=42
        )
        return df_sample

    except Exception as e:
        print("⚠ Stratified sampling failed → falling back to random sampling.", e)
        return df.sample(n=min(n, len(df)), random_state=42)


def process_language(lang):
    input_path = f"{INPUT_DIR}/{lang}.tsv"
    output_path = f"{OUTPUT_DIR}/{lang}_200.tsv"

    print(f"\n🔹 Processing language: {lang}")
    print(f"   Reading → {input_path}")

    df = pd.read_csv(input_path, sep="\t")

    # Perform stratified sampling
    df_sub = stratified_sample(df, n=200)

    # Save result
    df_sub.to_csv(output_path, sep="\t", index=False, encoding="utf-8")
    print(f"   Saved subset → {output_path}  ({len(df_sub)} samples)")


def main():
    print("=== Creating 200-sample stratified subsets for all languages ===")
    for lang in LANGUAGES:
        process_language(lang)

    print("\n🎉 Subset creation complete!")


if __name__ == "__main__":
    main()

