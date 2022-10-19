import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split

from src.main_contants import (
    UNUSED_COLS,
    TARGET,
    SEED,
    RANDOM_FOREST_ESTIMATORS,
    RAW_CSV_FILENAME,
)
from src.manage_dataframe import drop_columns, df_to_csv


def correlation_matrix(df: DataFrame):
    mat_corr = df.iloc[:, 1:-1].corr()
    mask = np.zeros_like(mat_corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(
        mat_corr, mask=mask, cmap="PuBu", vmin=-1, vmax=1, center=0, square=True
    )
    plt.show()
    df_to_csv(mat_corr, "../data/correlation_matrix.xlsx")


def evaluate(y_test, y_pred):
    return mse(y_test, y_pred) ** (1 / 2)


def plot_features_importance(features, columns):
    pd.Series(features, index=columns).sort_values().plot(
        kind="barh", color="lightgreen"
    )


def random_forest(df):
    # Drop les colonnes non numérique
    df = drop_columns(df, UNUSED_COLS)

    x = df.drop(TARGET, axis=1)
    y = df[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=SEED
    )

    # Instantiate a random forests regressor 'rf'
    rf = RandomForestRegressor(n_estimators=RANDOM_FOREST_ESTIMATORS, random_state=SEED)
    rf.fit(x_train, y_train)

    y_pred = rf.predict(x_test)

    # Evaluate the test
    print(f"Test set RMSE of rf: {evaluate(y_test, y_pred)}")

    # Create a pd.Series of features importances
    plot_features_importance(rf.feature_importances_, x.columns)
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(RAW_CSV_FILENAME)
    correlation_matrix(df)
    random_forest(df)
