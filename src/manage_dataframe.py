from typing import Iterable

import numpy as np
import pandas as pd
import ramda as R
from pandas import DataFrame

from main_contants import TARGET

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


"""
PIPES
"""


def clean_dataset(df: DataFrame) -> DataFrame:
    return R.pipe(
        drop_unused_columns,
        drop_duplicate_lines,
        df_dummies,
    )(df)


def read_clean_csv(filename_to_read: str, nrows: int | None = None):
    return R.pipe(read_csv(nrows), clean_dataset)(filename_to_read)


"""
GETTERS
"""


def get_zeros_matrix(matrix: Iterable):
    return np.zeros_like(matrix, dtype=bool)


def get_mask(matrix: DataFrame):
    mask = get_zeros_matrix(matrix)
    mask[np.triu_indices_from(mask)] = True
    return mask


def get_input(df: DataFrame):
    return df.drop(TARGET, axis=1)


def get_output(df: DataFrame):
    return df[TARGET]
