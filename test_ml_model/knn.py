from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from manage_dataframe import read_csv
from src.main_contants import RAW_CSV_FILENAME


def get_x_y(df):
    x_col = [
        "Source.Port",
        "Destination.Port",
        "Init_Win_bytes_forward",
        "Fwd.Packets.s",
    ]
    target = "L7Protocol"
    return df[x_col], df[target]


df = read_csv(None)(RAW_CSV_FILENAME)

df = df.loc[df["ProtocolName"].isin(["AMAZON", "MICROSOFT", "YOUTUBE", "GMAIL"])]
x, y = get_x_y(df)


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
