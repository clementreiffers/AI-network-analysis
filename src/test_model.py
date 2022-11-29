import pandas as pd
from keras import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.main_contants import TARGET


def get_x_y():
    df = pd.read_csv("../data/Dataset-Unicauca-Version2-87Atts.csv")
    x_col = [
        "Source.Port",
        "Destination.Port",
        "Flow.Duration",
        "Total.Length.of.Fwd.Packets",
        "Fwd.Packet.Length.Mean",
        "Fwd.Packet.Length.Std",
        "Flow.Bytes.s",
        "Flow.IAT.Max",
        "Flow.IAT.Min",
        "Fwd.IAT.Mean",
        "Fwd.IAT.Min",
        "Bwd.IAT.Total",
        "Bwd.IAT.Mean",
        "Bwd.IAT.Std",
        "Bwd.IAT.Max",
        "Fwd.Packets.s",
        "Bwd.Packets.s",
        "Avg.Fwd.Segment.Size",
        "Init_Win_bytes_forward",
        "Init_Win_bytes_backward",
        # "L7Protocol",
    ]
    return df[x_col], df[TARGET]


def create_model_from_layers(layers, mod):
    for layer in layers:
        mod.add(layer)
    return mod


def get_dl_model(x, y):
    layers = [
        Dense(32, input_shape=(len(x[0]),), activation="relu"),
        BatchNormalization(),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dense(512, activation="tanh"),
        BatchNormalization(),
        Dropout(0.20),
        Dense(len(y[0]), activation="softmax"),
    ]

    model = create_model_from_layers(layers, Sequential())
    model.compile(
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        optimizer="adam",
    )

    return model


if __name__ == "__main__":
    x, y = get_x_y()

    x = StandardScaler().fit_transform(x)
    y = to_categorical(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=1
    )

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

    # model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    #
    # model.fit(x_train, y_train)
    # y_pred = model.predict(x_test)
    # acc_dt = accuracy_score(y_test, y_pred)
    # print("df test accuracy : {:.4f}".format(acc_dt))
    model = get_dl_model(x, y)

    callbacks = [
        EarlyStopping(monitor="accuracy", patience=10),
        ModelCheckpoint("out/best.hdf5"),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        epochs=1000,
    )
    loss, acc = model.evaluate(x_test, y_test)
    print(f"loss:{loss}\nacc:{acc}")

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.legend(["train", "test"])
    plt.show()

    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.legend(["train", "test"])
    plt.show()
