import pandas as pd
from pandas import DataFrame

"""
DELETIONS
"""


def drop_unused_columns(df: DataFrame) -> DataFrame:
    return df.drop(df.columns[df.nunique() <= 1], axis=1)


def drop_columns(df: DataFrame, col: str | list[str]):
    return df.drop(columns=col, axis=1)


def drop_duplicate_lines(df: DataFrame) -> DataFrame:
    return df.drop_duplicates()


"""
LOADINGS
"""


def read_csv(nrows: int | None = None):
    def read(filename: str):
        return pd.read_csv(filename, nrows=nrows)

    return read


"""
SAVINGS
"""


def df_to_csv(df: DataFrame, filename: str):
    return df.to_csv(filename, index=False)


def df_to_xlsx(data, filename: str):
    return pd.DataFrame(data).to_excel(filename, index=False)


"""
ENCODERS
"""


def df_dummies(df: DataFrame):
    return pd.get_dummies(df)
