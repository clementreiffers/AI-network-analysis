import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.main_contants import TARGET


def base():
    df = pd.read_csv('../data/dataset_clean.csv')

    # list_colonne = ['Source.Port', 'Destination.Port', 'Flow.Duration', 'Bwd.Packet.Length.Max']
    list_protocol = ['AMAZON', 'YOUTUBE', 'MICROSOFT', 'GMAIL']

    df = df[df['ProtocolName'].isin(list_protocol)]
    df = df.drop(
        ["ProtocolName", "Month", "Day", "Protocol", "ECE.Flag.Count", "RST.Flag.Count", "Active.Max", "Active.Min",
         "Idle.Mean", "Idle.Max", "Active.Mean", "Idle.Std", "Bwd.Packet.Length.Min", "FIN.Flag.Count",
         "Min.Packet.Length", "Idle.Min", "Active.Std", "URG.Flag.Count", "SYN.Flag.Count", "ACK.Flag.Count",
         "Fwd.Packet.Length.Min", "Fwd.PSH.Flags", "PSH.Flag.Count", "Subflow.Fwd.Packets", "Total.Fwd.Packets",
         "Subflow.Bwd.Packets", "Subflow.Bwd.Bytes", "Total.Length.of.Bwd.Packets", "Down.Up.Ratio",
         "Bwd.Header.Length", "Packet.Length.Variance", "Packet.Length.Mean", "Subflow.Fwd.Bytes", "Max.Packet.Length",
         "Packet.Length.Std", "min_seg_size_forward", "Fwd.IAT.Max", "Average.Packet.Size", "Flow.IAT.Mean",
         "Flow.Packets.s", "Fwd.Packet.Length.Max", "act_data_pkt_fwd", "Fwd.Header.Length", "Fwd.Header.Length.1",
         "Avg.Bwd.Segment.Size", "Bwd.Packet.Length.Min", "Total.Backward.Packets", "Bwd.Packet.Length.Max",
         "Bwd.Packet.Length.Std", "Flow.IAT.Std", "Bwd.IAT.Min", "Fwd.IAT.Total", "Fwd.IAT.Std", "Flow.ID", "Source.IP",
         "Destination.IP", "Bwd.Packet.Length.Mean"], axis=1)
    col = df.columns

    y = df[TARGET]
    X = df[col]
    return y, X


y, X = base()

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=200)
# dt = DecisionTreeClassifier(max_depth=1, random_state=1)
# dt = DecisionTreeClassifier(random_state=42,max_depth=3, min_samples_leaf=5)


dt = DecisionTreeClassifier(max_depth=3,  random_state=1)
"""
# AVEC ca aussi on obtient accuracy = 1 !
dt = DecisionTreeClassifier(
    criterion='gini',
    splitter='best',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    random_state=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    class_weight=None,
    ccp_alpha=0.0
)"""

dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred)
print("df test accuracy : {:.4f}".format(acc_dt))



feature_names = list(X.columns)

from sklearn.tree import export_text
r = export_text(dt, feature_names=feature_names)
print(r)

# df test accuracy : 0.7280
# 0.7307 avec random_state = 100
# 0.7345


