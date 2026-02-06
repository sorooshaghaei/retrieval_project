"""Intuition: Search engines don't look at "Title" and "Text" separately usually. 
They look at one big blob of text. We need to combine: Title + Text + Tags = Content. 
Also, we should lowercase everything"""

import pandas as pd


def clean_text(text):
    """
    1. checks if text is not empty --> isna is from pandas.
        isna = is_not_available
    2. convert to lowercase
    """
    if pd.isna(text) or text == "":
        return ""
    return str(text).lower()


def safe_value_to_string(val):
    """
    Helper: Converts a value to a string.
    If it's a list (like tags), joins them with spaces.
    Example: ['a', 'b'] -> "a b"
    """
    if isinstance(val, list):
        return " ".join(str(v) for v in val)
    return str(val)


def create_content_column(df, columns_to_merge):
    """
    creates a single 'content' column for the search engine 

        arguments:
            df (DataFrame): The dataframe (docs or queries)
            columns_to_merge (list): List of column names to combine (e.g., ['title', 'text'])

        Returns:
            DataFrame: The dataframe with a new 'content' column
    """
    print("creating 'content' column...")
    df_clean = df.copy()

    # 1. fill up empty values to avoid crashes
    for col in columns_to_merge:
        # if col exist fill NaN
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
