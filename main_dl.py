import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.utils import to_categorical
import warnings
warnings.filterwarnings("ignore")

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV


df = pd.read_csv('/Users/maellemarcelin/Documents/sudria/Ingé3/projet/projetIA/projetIA/3- Dataset/dataset_clean.csv',sep=',',encoding='utf8')

df_select = df.drop(["Protocol","ECE.Flag.Count","RST.Flag.Count","Active.Max","Active.Min","Idle.Mean","Idle.Max","Active.Mean","Idle.Std","Bwd.Packet.Length.Min","FIN.Flag.Count","Min.Packet.Length","Idle.Min","Active.Std","URG.Flag.Count","SYN.Flag.Count","ACK.Flag.Count","Fwd.Packet.Length.Min","Fwd.PSH.Flags","PSH.Flag.Count","Subflow.Fwd.Packets","Total.Fwd.Packets","Subflow.Bwd.Packets","Subflow.Bwd.Bytes","Total.Length.of.Bwd.Packets","Down.Up.Ratio","Bwd.Header.Length","Packet.Length.Variance","Packet.Length.Mean","Subflow.Fwd.Bytes","Max.Packet.Length","Packet.Length.Std","min_seg_size_forward","Fwd.IAT.Max","Average.Packet.Size","Flow.IAT.Mean","Flow.Packets.s","Fwd.Packet.Length.Max","act_data_pkt_fwd", "Fwd.Header.Length", "Fwd.Header.Length.1","Avg.Bwd.Segment.Size","Bwd.Packet.Length.Min","Total.Backward.Packets","Bwd.Packet.Length.Max","Bwd.Packet.Length.Std","Flow.IAT.Std","Bwd.IAT.Min","Fwd.IAT.Total","Fwd.IAT.Std","Flow.ID","Source.IP","Destination.IP","Timestamp","Bwd.Packet.Length.Mean"], axis=1)

df_reduce = df_select[(df_select['ProtocolName'] == 'AMAZON') | (df_select['ProtocolName'] == 'YOUTUBE') | (df_select['ProtocolName'] == 'MICROSOFT') | (df_select['ProtocolName'] == 'GMAIL') | (df_select['ProtocolName'] == 'WINDOWS_UPDATE') | (df_select['ProtocolName'] == 'SKYPE') | (df_select['ProtocolName'] == 'FACEBOOK') | (df_select['ProtocolName'] == 'DROPBOX')]

X = df_reduce.drop(['ProtocolName',"L7Protocol"], axis=1)
y = df_reduce['L7Protocol']
y = to_categorical(y)


## modèle de base pour la recherche des paramètres optimaux

def create_model(activation='relu', number_of_neurons=256, number_of_layers=2, init_mode='uniform'):
    model = Sequential()
    model.add(Dense(128, input_shape=(20,), kernel_initializer=init_mode, activation=activation))
    for i in range(number_of_layers):
        model.add(Dense(number_of_neurons, activation=activation))
    model.add(Dense(213, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


skmodel = KerasClassifier(build_fn=create_model)

params = {'activation':['relu', 'tanh'],
          'batch_size':[4, 8, 16, 32, 64],
          'epochs':[20, 50, 100, 200, 500],
          'init_mode':['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'],
          'number_of_layers':[3,5,7,9,11],
          'number_of_neurons':[8,16,32,64,128,256]
}

random_search = RandomizedSearchCV(skmodel, param_distributions=params, cv=3, n_jobs=-1)

random_search_results = random_search.fit(X, y, verbose=0)

file = open("result.txt", "w")
file.write("Best: {} using {}".format(np.round(random_search_results.best_score_,4), random_search_results.best_params_))
file.close()
