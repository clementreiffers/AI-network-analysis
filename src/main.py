from manage_ml import (
    correlation_matrix,
    read_clean_csv,
    separate_and_random_forest,
)
from main_contants import RAW_CSV_FILENAME, FILENAME_CLEANED_DATA
from manage_dataframe import df_to_csv, get_input, get_output

if __name__ == "__main__":
    df = read_clean_csv(RAW_CSV_FILENAME, nrows=1000)
    df_to_csv(df, FILENAME_CLEANED_DATA)

    x = get_input(df)
    y = get_output(df)

    correlation_matrix(df)
    separate_and_random_forest(x, y, 1000)
