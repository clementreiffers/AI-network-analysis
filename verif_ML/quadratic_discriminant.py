from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


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

##df_reduce = df_select[(df_select['ProtocolName'] == 'AMAZON') | (df_select['ProtocolName'] == 'YOUTUBE') | (df_select['ProtocolName'] == 'MICROSOFT') | (df_select['ProtocolName'] == 'GMAIL') | (df_select['ProtocolName'] == 'WINDOWS_UPDATE') | (df_select['ProtocolName'] == 'SKYPE') | (df_select['ProtocolName'] == 'FACEBOOK') | (df_select['ProtocolName'] == 'DROPBOX')]


df_encode = df_reduce.copy()


y = df_encode['L7Protocol']
X = df_encode.drop(['ProtocolName',"L7Protocol"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)


clf = QuadraticDiscriminantAnalysis()
clf.fit(X, y)

y_pred = clf.predict(X_test)

print("Accuracy sur les données de test: {0:0.4f}".format(accuracy_score(y_test, y_pred)))