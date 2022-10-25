from main_contants import RAW_CSV_FILENAME, FILENAME_CLEANED_DATA, STATISTICS_CSV
from manage_dataframe import (
    df_to_csv,
    read_clean_csv,
    get_stats_from_dataframe,
    df_stats_to_csv,
)
from manage_ml import correlation_matrix

if __name__ == "__main__":
    df = read_clean_csv(RAW_CSV_FILENAME, nrows=1000)
    df_to_csv(df, FILENAME_CLEANED_DATA)

    correlation_matrix(df)
    df_stats_to_csv(df, STATISTICS_CSV)
    # x = get_input(df)
    # y = get_output(df)
    # separate_and_random_forest(x, y, 1000)
