import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import SGDClassifier, ElasticNet, BayesianRidge
from sklearn.multiclass import OutputCodeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.main_contants import TARGET


def base():
    df = pd.read_csv("../data/dataset_clean.csv")
    # list_protocol = ['AMAZON', 'YOUTUBE', 'MICROSOFT', 'GMAIL']
    # list_protocol = ["AMAZON", "MICROSOFT", "YOUTUBE", "GMAIL", "WINDOWS_UPDATE", "SKYPE", "FACEBOOK", "DROPBOX"]
    # df = df[df['ProtocolName'].isin(list_protocol)]
    df = df.drop(
        [
            "ProtocolName",
            "Month",
            "Day",
            "Protocol",
            "ECE.Flag.Count",
            "RST.Flag.Count",
            "Active.Max",
            "Active.Min",
            "Idle.Mean",
            "Idle.Max",
            "Active.Mean",
            "Idle.Std",
            "Bwd.Packet.Length.Min",
            "FIN.Flag.Count",
            "Min.Packet.Length",
            "Idle.Min",
            "Active.Std",
            "URG.Flag.Count",
            "SYN.Flag.Count",
            "ACK.Flag.Count",
            "Fwd.Packet.Length.Min",
            "Fwd.PSH.Flags",
            "PSH.Flag.Count",
            "Subflow.Fwd.Packets",
            "Total.Fwd.Packets",
            "Subflow.Bwd.Packets",
            "Subflow.Bwd.Bytes",
            "Total.Length.of.Bwd.Packets",
            "Down.Up.Ratio",
            "Bwd.Header.Length",
            "Packet.Length.Variance",
            "Packet.Length.Mean",
            "Subflow.Fwd.Bytes",
            "Max.Packet.Length",
            "Packet.Length.Std",
            "min_seg_size_forward",
            "Fwd.IAT.Max",
            "Average.Packet.Size",
            "Flow.IAT.Mean",
            "Flow.Packets.s",
            "Fwd.Packet.Length.Max",
            "act_data_pkt_fwd",
            "Fwd.Header.Length",
            "Fwd.Header.Length.1",
            "Avg.Bwd.Segment.Size",
            "Bwd.Packet.Length.Min",
            "Total.Backward.Packets",
            "Bwd.Packet.Length.Max",
            "Bwd.Packet.Length.Std",
            "Flow.IAT.Std",
            "Bwd.IAT.Min",
            "Fwd.IAT.Total",
            "Fwd.IAT.Std",
            "Flow.ID",
            "Source.IP",
            "Destination.IP",
            "Bwd.Packet.Length.Mean",
        ],
        axis=1,
    )
    col = df.columns

    y = df[TARGET]
    X = df[col]
    return y, X


y, X = base()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

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

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred)
print("df test accuracy : {:.4f}".format(acc_dt))
