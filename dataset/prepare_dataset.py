import os
import re
import string
import random
import json
import pandas as pd
import numpy as np
from collections import Counter


def load_raw(filepath: str) -> pd.DataFrame:
    """Load JSON or CSV dataset."""
    if filepath.endswith('.json') or filepath.endswith('.jsonl'):
        df = pd.read_json(filepath, lines=True)
    else:
        df = pd.read_csv(filepath)

    if 'headline' in df.columns and 'short_description' in df.columns:
        df['text'] = df['headline'].fillna('') + ' ' + df['short_description'].fillna('')
    elif 'text' not in df.columns:
        raise ValueError("Dataset needs 'text' or 'headline'+'short_description' columns.")

    df = df[['text', 'category']].dropna()
    df['text'] = df['text'].str.strip()
    df = df[df['text'].str.len() > 20] 
    return df

def analyze_distribution(df: pd.DataFrame) -> dict:
    """Print and return class distribution statistics."""
    counts = df['category'].value_counts()
    total = len(df)

    print("\nClass Distribution:")
    print(f"{'Category':<35} {'Count':>8} {'%':>8}")
    print("-" * 55)
    for cat, cnt in counts.items():
        print(f"{cat:<35} {cnt:>8,} {cnt/total*100:>7.1f}%")

    imbalance_ratio = counts.max() / counts.min()
    print(f"\nImbalance Ratio (max/min): {imbalance_ratio:.1f}x")
    if imbalance_ratio > 5:
        print("   High imbalance detected — augmentation recommended.")

    return counts.to_dict()


def synonym_swap(text: str, swap_prob: float = 0.15) -> str:
    """
    Simple word dropout augmentation.
    Randomly drops words to create slightly different training samples.
    (Use nlpaug for proper synonym replacement in production.)
    """
    words = text.split()
    if len(words) < 5:
        return text
    augmented = [w for w in words if random.random() > swap_prob]
    return ' '.join(augmented) if len(augmented) >= 3 else text


def random_word_insertion(text: str) -> str:
    
    words = text.split()
    if len(words) < 4:
        return text
   
    content_words = [w for w in words if len(w) > 4]
    if not content_words:
        return text
    insert_word = random.choice(content_words)
    pos = random.randint(0, len(words))
    words.insert(pos, insert_word)
    return ' '.join(words)


def augment_minority_classes(df: pd.DataFrame, min_samples: int = 500) -> pd.DataFrame:
    """
    Augment categories with fewer than `min_samples` examples.
    Uses multiple augmentation strategies.
    """
    counts = df['category'].value_counts()
    minority_cats = counts[counts < min_samples].index.tolist()

    if not minority_cats:
        print(" No minority classes detected — no augmentation needed.")
        return df

    print(f"\n🔧 Augmenting {len(minority_cats)} minority categories...")
    augmented_rows = []

    for cat in minority_cats:
        cat_df = df[df['category'] == cat]
        needed = min_samples - len(cat_df)

        for _ in range(needed):
            row = cat_df.sample(1).iloc[0]
           
            strategy = random.choice(['dropout', 'insertion', 'both'])
            text = row['text']
            if strategy == 'dropout':
                text = synonym_swap(text)
            elif strategy == 'insertion':
                text = random_word_insertion(text)
            else:
                text = random_word_insertion(synonym_swap(text))

            augmented_rows.append({'text': text, 'category': cat})

        print(f"  {cat}: {len(cat_df)} → {min_samples} (+{needed} samples)")

    aug_df = pd.DataFrame(augmented_rows)
    result = pd.concat([df, aug_df], ignore_index=True).sample(frac=1, random_state=42)
    return result


def stratified_split(df: pd.DataFrame, val_size: float = 0.1, test_size: float = 0.1):
    """Stratified split into train, validation, and test sets."""
    from sklearn.model_selection import train_test_split

    X = df['text'].values
    y = df['category'].values

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac, stratify=y_trainval, random_state=42
    )

    print(f"\nSplit sizes — Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    return (
        pd.DataFrame({'text': X_train, 'category': y_train}),
        pd.DataFrame({'text': X_val,   'category': y_val}),
        pd.DataFrame({'text': X_test,  'category': y_test})
    )

if __name__ == '__main__':
    import sys

    raw_path  = sys.argv[1] if len(sys.argv) > 1 else 'news.json'
    out_dir   = sys.argv[2] if len(sys.argv) > 2 else '.'

    print(f"Loading: {raw_path}")
    df = load_raw(raw_path)
    print(f"{len(df):,} samples | {df['category'].nunique()} categories")

    analyze_distribution(df)
    df = augment_minority_classes(df, min_samples=500)

    train_df, val_df, test_df = stratified_split(df)

    os.makedirs(out_dir, exist_ok=True)
    train_df.to_csv(f'{out_dir}/train.csv', index=False)
    val_df.to_csv(f'{out_dir}/val.csv',   index=False)
    test_df.to_csv(f'{out_dir}/test.csv', index=False)

    print(f"\nSaved train/val/test CSVs to '{out_dir}/'")
