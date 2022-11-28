# load the model from disk
import pickle

from sklearn.model_selection import train_test_split

from src.decisionTree import base


def load_model(filename, X_test, Y_test):
    loaded_model = pickle.load(open(filename, "rb"))
    result = loaded_model.score(X_test, Y_test)
    print(result)


filename = "finalized_model.sav"


y, X = base()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=200
)

load_model(filename, X_test, y_test)
