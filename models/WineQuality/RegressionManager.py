from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge

class RegressionManager:
    def __init__(self, dataframe):
        self._dataframe = dataframe
        self._models = {}
        self._evaluations = {}

    def addModel(self, key, model, features_list, label):
        X = self._dataframe[features_list]
        Y = self._dataframe[label]
        x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)
        key = key + " using features: " + " ".join(features_list)
        self._models[key] = model
        self._evaluations[key] = {}
        self._evaluations[key]['training_results'] = self._train_model(model, x_train, y_train)
        self._evaluations[key]['test_results'] = self._test_model(model, x_test, y_test)
        plt.title(key)

    def _train_model(self, model: LinearRegression, x_train, y_train):
        model.fit(x_train, y_train)
        return {'score': model.score(x_train, y_train)}

    def _test_model(self, model: LinearRegression, x_test, y_test):
        y_pred = model.predict(x_test)
        plt.figure()
        plt.plot(y_pred, label='Predicted')
        plt.plot(y_test.values, label='Actual')
        plt.legend()
        return {'score': r2_score(y_test, y_pred)}

    def showEvaluations(self):
        for model_key in self._evaluations:
            print(model_key)
            print()
            print("Training result score: " + str(self._evaluations[model_key]['training_results']['score']))
            print("Test result score: " + str(self._evaluations[model_key]['test_results']['score']))
            print()