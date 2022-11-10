from main_contants import RAW_CSV_FILENAME
from manage_dataframe import read_csv, get_list_of_dict_key_value_len

if __name__ == "__main__":
    # df = read_clean_csv(RAW_CSV_FILENAME, nrows=1000)
    # df_to_csv(df, FILENAME_CLEANED_DATA)

    df = read_csv(None)(RAW_CSV_FILENAME)
    a = sorted(
        get_list_of_dict_key_value_len(df, "L7Protocol"),
        key=lambda data: data["nbr"],
    )
    a.reverse()
    print(a)
    print(len(a))
    # x = get_input(df)
    # y = get_output(df)

    # correlation_matrix(df)
    # df_stats_to_csv(df, STATISTICS_CSV)
    #
    # g = get_pair_grid(df)
    # g.map_diag(plt.hist, histtype="step", linewidth=3)
    # g.map_offdiag(plt.scatter)
    # plt.show()

    # separate_and_random_forest(x, y, 1000)
#
