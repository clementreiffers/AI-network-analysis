import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE


def correlation_matrix(df):
    mat_corr = df.iloc[:, 1:-1].corr()
    mask = np.zeros_like(mat_corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(mat_corr, mask=mask, cmap="PuBu", vmin=-1, vmax=1, center=0, square=True)
    plt.show()
    # save_matrix_in_excel(mat_corr)


def save_matrix_in_excel(mat_corr):
    df = pd.DataFrame(mat_corr)
    df.to_excel('correlation_matrix.xlsx', index=False)


def random_forest(df):
    # Drop les colonnes non numérique
    df = df.drop(columns=['Flow.ID', 'Timestamp', 'ProtocolName', 'Source.IP', 'Destination.IP'])

    # Réduction taille df pour aller + vite
    # df = df[:10000]

    X = df.drop('L7Protocol', axis=1)
    y = df['L7Protocol']

    # Set seed for reproducibility
    SEED = 1

    # Split dataset into 70% train and 30% test
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=SEED)

    # Instantiate a random forests regressor 'rf'
    rf = RandomForestRegressor(n_estimators=50,
                               # min_samples_leaf=0.12,
                               random_state=SEED)
    # Fit 'rf' to the training set
    rf.fit(X_train, y_train)
    # Predict the test set labels 'y_pred'
    y_pred = rf.predict(X_test)

    # Evaluate the test
    rmse_test = MSE(y_test, y_pred) ** (1 / 2)
    print('Test set RMSE of rf: {:.2f}'.format(rmse_test))

    # Create a pd.Series of features importances
    importances_rf = pd.Series(rf.feature_importances_, index=X.columns)
    sorted_importances_rf = importances_rf.sort_values()
    sorted_importances_rf.plot(kind='barh', color='lightgreen')
    plt.show()

    return None


df = pd.read_csv('../dataset_clean.csv')
# correlation_matrix(df)
random_forest(df)

# Test set RMSE of rf: 37.08 mini test
# Test set RMSE of rf: 22.46 sur 1000
# Test set RMSE of rf: 26.51 sur 10000












