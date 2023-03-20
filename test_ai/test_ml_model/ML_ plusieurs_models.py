import pandas as pd
from keras.optimizers import SGD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
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
    df = pd.read_csv("../../data/dataset_clean.csv")

    # list_protocol = ['AMAZON', 'YOUTUBE', 'MICROSOFT', 'GMAIL']
    list_protocol = ["AMAZON", "MICROSOFT", "YOUTUBE", "GMAIL", "WINDOWS_UPDATE", "SKYPE", "FACEBOOK", "DROPBOX"]
    df = df[df['ProtocolName'].isin(list_protocol)]
    X = df[['Source.Port', 'Destination.Port', 'Flow.Duration',
            'Total.Length.of.Fwd.Packets', 'Fwd.Packet.Length.Mean',
            'Fwd.Packet.Length.Std', 'Flow.Bytes.s', 'Flow.IAT.Max', 'Flow.IAT.Min',
            'Fwd.IAT.Mean', 'Fwd.IAT.Min', 'Bwd.IAT.Total', 'Bwd.IAT.Mean',
            'Bwd.IAT.Std', 'Bwd.IAT.Max', 'Fwd.Packets.s', 'Bwd.Packets.s',
            'Avg.Fwd.Segment.Size', 'Init_Win_bytes_forward',
            'Init_Win_bytes_backward']]
    y = df[TARGET]

    return y, X


y, X = base()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# model = RandomForestClassifier(n_estimators=25, random_state=1)
# model = SVC()
# model = SGDClassifier(loss="hinge", penalty="l2") # 0.4062
# model = MLPClassifier(hidden_layer_sizes=(30,), random_state=1, warm_start=True) # 0.5200
# model = LinearDiscriminantAnalysis()
# model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
# model = make_pipeline(StandardScaler(), SGDClassifier(loss="hinge", penalty="l2"))
# model = make_pipeline(StandardScaler(), GaussianNB())
model = DecisionTreeClassifier(max_depth=20)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred)
print("df test accuracy : {:.4f}".format(acc_dt))
