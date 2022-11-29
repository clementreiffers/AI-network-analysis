import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.main_contants import TARGET


def base():
    df = pd.read_csv("../data/dataset_clean.csv")
    x_col = [
        "Source.Port",
        "Destination.Port",
        "Init_Win_bytes_forward",
        "Fwd.Packets.s",
        "Init_Win_bytes_backward",
        "Bwd.Packets.s",
        "Flow.Bytes.s",
        "Avg.Fwd.Segment.Size",
        "Fwd.Packet.Length.Std",
        "Flow.IAT.Max",
        "Flow.IAT.Min",
        "Bwd.IAT.Total",
        "Fwd.IAT.Mean",
        "Bwd.IAT.Std",
        "Bwd.IAT.Max",
        "Bwd.IAT.Mean",
        "Total.Length.of.Fwd.Packets",
        "Fwd.IAT.Total",
        "act_data_pkt_fwd",
        "Fwd.Packet.Length.Max",
    ]
    return df[TARGET], df[x_col]


y, X = base()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

"""
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
model = GaussianNB()
"""


# model = SVC()
# model = SGDClassifier(loss="hinge", penalty="l2") # 0.4062
# model = MLPClassifier(hidden_layer_sizes=(30,), random_state=1, warm_start=True) # 0.5200
# model = LinearDiscriminantAnalysis()
# model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
# model = make_pipeline(StandardScaler(), SGDClassifier(loss="hinge", penalty="l2"))
# model = make_pipeline(StandardScaler(), GaussianNB())

model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
acc_dt = accuracy_score(y_test, y_pred)
print("df test accuracy : {:.4f}".format(acc_dt))
