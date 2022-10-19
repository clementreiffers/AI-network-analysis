import pandas as pd
from pandas import DataFrame
import ramda as R


"""
DELETIONS
"""


def drop_unused_columns(df: DataFrame) -> DataFrame:
    return df.drop(df.columns[df.nunique() <= 1], axis=1)


def drop_columns(df: DataFrame, col: str | list[str]):
    return df.drop(columns=col, axis=1)


def drop_duplicate_lines(df: DataFrame) -> DataFrame:
    return df.drop_duplicates()


def clean_dataset(df: DataFrame) -> DataFrame:
    return R.pipe(drop_unused_columns, drop_duplicate_lines)(df)


"""
LOADINGS
"""


def read_csv(filename: str):
    return pd.read_csv(filename)


"""
SAVINGS
"""


def df_to_csv(df: DataFrame, filename: str):
    return df.to_csv(filename, index=False)


def df_to_xlsx(data, filename: str):
    return pd.DataFrame(data).to_excel(filename, index=False)
