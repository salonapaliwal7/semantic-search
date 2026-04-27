# preprocess.py
"""
Handles dataset loading and text preprocessing.
"""


def combine_fields(row):
    title = str(row.get('title', row.get('name', '')))
    desc = str(row.get('description', row.get('features', '')))
    brand = str(row.get('brand', ''))
    cat = str(row.get('main_category', row.get('category', '')))
    return f"{title}. {desc}. Brand: {brand}. Category: {cat}."

def preprocess_data(df):
    # Convert Hugging Face DatasetDict -> Dataset -> pandas.
    if not hasattr(df, "to_pandas") and hasattr(df, "keys"):
        splits = list(df.keys())
        if splits:
            split_name = "train" if "train" in splits else splits[0]
            df = df[split_name]

    # Convert Hugging Face Dataset to pandas for downstream pipeline use.
    if hasattr(df, "to_pandas"):
        df = df.to_pandas()

    # Build combined text row-wise.
    df["combined_text"] = df.apply(combine_fields, axis=1)
    return df
