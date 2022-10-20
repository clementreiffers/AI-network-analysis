import matplotlib.pyplot as plt
import pandas as pd
import ramda as R
import seaborn as sns
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split

from main_contants import SEED, RANDOM_FOREST_ESTIMATORS, TARGET
from manage_dataframe import df_to_csv, get_mask
from manage_list import truncate


def show_heatmap(matrix: DataFrame):
    sns.heatmap(
        matrix,
        mask=get_mask(matrix),
        cmap="PuBu",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
    )
    plt.show()


def correlation_matrix(df: DataFrame):
    mat_corr = df.iloc[:, 1:-1].corr()
    show_heatmap(mat_corr)
    df_to_csv(mat_corr, "../data/correlation_matrix.xlsx")


def evaluate(y_test: list[int], y_pred: list[int]) -> float:
    return mse(y_test, y_pred) ** (1 / 2)


def get_features_importance(features: list, columns: list):
    return pd.Series(features, index=columns).sort_values()


def show_features_importance(features: list, columns: list):
    get_features_importance(features, columns).plot(kind="barh", color="lightgreen")
    plt.show()


def separate_data(x: DataFrame, y: DataFrame):
    return train_test_split(x, y, test_size=0.3, random_state=SEED)


def get_model():
    return RandomForestRegressor(
        n_estimators=RANDOM_FOREST_ESTIMATORS, random_state=SEED
    )


def random_forest(max_rank: int | None = None):
    def process_rf(data: list[DataFrame]):
        x_train, x_test, y_train, y_test = data
        # Instantiate a random forests regressor 'rf'
        rf = get_model()
        rf.fit(x_train, y_train)

        y_pred = rf.predict(x_test)

        # Evaluate the test
        print(f"Test set RMSE of rf: {evaluate(y_test, y_pred)}")

        # Create a pd.Series of features importances
        features = truncate(rf.feature_importances_, max_rank)
        columns = truncate(x_train.columns, max_rank)

        show_features_importance(features, columns)
        return rf

    return process_rf


def separate_and_random_forest(x: DataFrame, y: DataFrame, max_rank: int | None = None):
    R.pipe(separate_data, random_forest(max_rank))(x, y)
