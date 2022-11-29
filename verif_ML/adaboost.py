import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('/Users/maellemarcelin/Documents/sudria/Ingé3/projet/projetIA/projetIA/3- Dataset/dataset_clean.csv',
                 sep=',', encoding='utf8')


df_select = df[['Init_Win_bytes_backward', 'Bwd.Packets.s', 'Flow.Bytes.s',
       'Fwd.Packet.Length.Std', 'Fwd.Packet.Length.Mean',
       'Flow.IAT.Min', 'Bwd.IAT.Std', 'Total.Length.of.Fwd.Packets',
        'Source.Port', 'Destination.Port', 'Flow.Duration'
        ,'Flow.IAT.Max', 'Flow.IAT.Min','Fwd.IAT.Mean',
        'Fwd.IAT.Min','Fwd.IAT.Mean', 'Bwd.IAT.Total',
        'Bwd.IAT.Mean', 'Bwd.IAT.Max', 'Fwd.Packets.s',
        'Avg.Fwd.Segment.Size','Init_Win_bytes_forward', 'ProtocolName', 'L7Protocol']]

df_reduce = df_select[(df_select['ProtocolName'] == 'AMAZON') | (df_select['ProtocolName'] == 'YOUTUBE') | (
            df_select['ProtocolName'] == 'MICROSOFT') | (df_select['ProtocolName'] == 'GMAIL')]

df_encode = df_reduce.copy()

print(df_encode.columns)

y = df_encode['L7Protocol']
X = df_encode.drop(['ProtocolName',"L7Protocol"], axis=1)

SEED = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)

dt = DecisionTreeClassifier(max_depth=1, random_state=SEED)

adb_clf = AdaBoostClassifier(base_estimator=dt, n_estimators=200)

adb_clf.fit(X_train, y_train)

AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1,
                                                         random_state=1),
                   n_estimators=200)

y_pred = adb_clf.predict(X_test)
print('adaboost(DT)')
print("Accuracy sur les données de test: {0:0.4f}".format(accuracy_score(y_test, y_pred)))

from sklearn.ensemble import RandomForestClassifier


rf = RandomForestClassifier(random_state=1, n_estimators=10)

adb_clf = AdaBoostClassifier(base_estimator=rf, n_estimators=100)

adb_clf.fit(X_train, y_train)

AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=10,
                                                         random_state=1),
                   n_estimators=100)

y_pred = adb_clf.predict(X_test)

print("adaboost(random forest)")
print("Accuracy sur les données de test: {0:0.4f}".format(accuracy_score(y_test, y_pred)))

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

scaler = StandardScaler()
pca = PCA()
rf2 = RandomForestClassifier(random_state=1, n_estimators=10)
adb_clf2 = AdaBoostClassifier(base_estimator=rf2, n_estimators=100)

pipeline = make_pipeline(scaler, pca, adb_clf2)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("standard scaler + adaboost(random forest)")

print('score: {}'.format(pipeline.score(X_test, y_test)))
print("Accuracy sur les données de test: {0:0.4f}".format(accuracy_score(y_test, y_pred)))

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

scaler = StandardScaler()
pca = PCA()
dt2 = DecisionTreeClassifier(max_depth=1, random_state=SEED)
adb_clf2 = AdaBoostClassifier(base_estimator=dt2, n_estimators=100)

pipeline = make_pipeline(scaler, pca, adb_clf2)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)


print("standard scaler + adaboost(DT)")
print('score: {}'.format(pipeline.score(X_test, y_test)))

print("Accuracy sur les données de test: {0:0.4f}".format(accuracy_score(y_test, y_pred)))

