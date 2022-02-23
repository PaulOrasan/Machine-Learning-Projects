from Dataset import Dataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

class Classifier:
    def __init__(self, key, model, train_dataset: Dataset, features, label):
        self._name = key
        self._model = model
        self._features = features
        self._label = label
        self._evaluation = {}
        self._train(train_dataset, features, label)

    def _train(self, train_dataset: Dataset, features, label):
        x_train, y_train = train_dataset.splitXY(features, label)
        print(x_train.head())
        print(y_train.head())
        self._model.fit(x_train, y_train)
        y_pred = self._model.predict(x_train)
        self._evaluation['training'] = {'accuracy': accuracy_score(y_train, y_pred),
                                        'precision': precision_score(y_train, y_pred),
                                        'recall': recall_score(y_train, y_pred)}

    def test(self, test_dataset: Dataset):
        x_test, y_test = test_dataset.splitXY(self._features, self._label)
        y_pred = self._model.predict(x_test)
        self._evaluation['testing'] = {'accuracy': accuracy_score(y_test, y_pred),
                                        'precision': precision_score(y_test, y_pred),
                                        'recall': recall_score(y_test, y_pred)}

    def getEvaluations(self):
        return self._evaluation

    def __str__(self):
        return self._name

