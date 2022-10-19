from src.main_contants import RAW_CSV_FILENAME, FILENAME_CLEANED_DATA
from src.manage_dataframe import clean_dataset, read_csv, df_to_csv
import ramda as R


def read_clean_csv(filename_to_read: str):
    return R.pipe(read_csv, clean_dataset)(filename_to_read)


if __name__ == "__main__":
    df = read_clean_csv(RAW_CSV_FILENAME)
    df_to_csv(df, FILENAME_CLEANED_DATA)
