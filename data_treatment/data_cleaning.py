import pandas as pd


def drop_unused_columns(df):
    unused_columns = df.columns[df.nunique() <= 1]
    return df.drop(unused_columns, axis=1)


def drop_duplicate_lines(df):
    return df.drop_duplicates()


def clean_dataset(df):
    df = drop_unused_columns(df)
    df = drop_duplicate_lines(df)
    return df


df = pd.read_csv('../dataset.csv')
df = clean_dataset(df)
df.to_csv('../dataset_clean.csv', index=False)
