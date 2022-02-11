from sklearn.model_selection import train_test_split

class DataManager:
    def __init__(self, X, Y):
        self._x_train, self._x_test, self._y_train, self._y_test = train_test_split(X, Y, train_size=0.8)

    def getXTrain(self):
        return self._x_train

    def getYTrain(self):
        return self._y_train

    def getXTest(self):
        return self._x_test

    def getYTest(self):
        return self._y_test