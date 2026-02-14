"""Search engines don't look at "Title" and "Text" separately usually. 
They look at one big blob of text. We need to combine: Title + Text + Tags = Content. 
Also, we should lowercase everything"""

import pandas as pd


def clean_text(text):
    if pd.isna(text) or text == "":
        return ""
    return str(text).lower()


def safe_value_to_string(val):
    if isinstance(val, list):
        return " ".join(str(v) for v in val)
    return str(val)


def create_content_column(df, columns_to_merge):
    print("creating 'content' column...")
    df_clean = df.copy()

    # 1. fill up empty values to avoid crashes -> if col exist fill NaN
    for col in columns_to_merge:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna("")
        else:
            # if a column (like tags) is missing in queries, create it as empty
            df_clean[col] = ""

    # 2. Combine columns safely (Handling lists like 'tags')
    contents = []

    for i in range(len(df_clean)):
        parts = []
        for col in columns_to_merge:
            parts.append(safe_value_to_string(df_clean.at[i, col]))
        contents.append(" ".join(parts))
    df_clean["content"] = contents

    df_clean["content"] = [clean_text(text) for text in df_clean["content"]]

    return df_clean
