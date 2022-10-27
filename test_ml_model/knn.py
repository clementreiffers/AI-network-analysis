from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from constants import X_COL, TARGET
from main_contants import RAW_CSV_FILENAME
from manage_dataframe import read_clean_csv, read_csv, df_dummies


def get_x_y(df):
    return df[X_COL], df[TARGET]


df = read_csv(None)(RAW_CSV_FILENAME)

df = df.loc[df["ProtocolName"].isin(["AMAZON", "MICROSOFT", "YOUTUBE", "GMAIL"])]
x, y = get_x_y(df)

x = df_dummies(x)
y = df_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    shuffle=True,
    random_state=0,
)

scaler = StandardScaler()
pca = PCA()
knn = KNeighborsClassifier(4)
knn2 = KNeighborsClassifier(2)

pipeline = make_pipeline(scaler, pca, knn)

pipeline.fit(x_train, y_train)
knn2.fit(x_train, y_train)

print("--- TRAIN ---")
print("avec standardisation et pca : ", pipeline.score(x_train, y_train))
print("knn seul : ", knn2.score(x_train, y_train))

print("--- TEST ---")
print("avec standardisation et pca : ", pipeline.score(x_test, y_test))
print("knn seul : ", knn2.score(x_test, y_test))
