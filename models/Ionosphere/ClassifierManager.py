from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from colorama import Fore

from DataManager import DataManager

class ClassifierManager:
    def __init__(self, X, Y):
        self._data = DataManager(X, Y)
        self._classifierEvaluation = {}
        self._classifiers = {}

    def addClassifier(self, classifier_name, classifier):
        self._classifiers[classifier_name] = classifier
        self._classifierEvaluation[classifier_name] = self._evaluateClassifier(classifier)

    def _trainClassifier(self, classifier):
        classifier.fit(self._data.getXTrain(), self._data.getYTrain())

    def _testClassifier(self, classifier, x_test, y_test):
        y_pred = classifier.predict(x_test)
        results = {}
        results['accuracy'] = accuracy_score(y_test, y_pred, normalize=True)
        results['accuracy_count'] = accuracy_score(y_test, y_pred, normalize=False)
        results['precision'] = precision_score(y_test, y_pred)
        results['recall'] = recall_score(y_test, y_pred)
        return results

    def _evaluateClassifier(self, classifier):
        self._trainClassifier(classifier)
        results = {}
        results['train_results'] = self._testClassifier(classifier, self._data.getXTrain(), self._data.getYTrain())
        results['test_results'] = self._testClassifier(classifier, self._data.getXTest(), self._data.getYTest())
        return results

    def showEvaluations(self):
        for classifier_name in self._classifierEvaluation:
            print(Fore.RED + classifier_name)
            print(Fore.WHITE)
            print("Training results: ")
            for key in self._classifierEvaluation[classifier_name]['train_results']:
                print(Fore.YELLOW + key, self._classifierEvaluation[classifier_name]['train_results'][key])
            print(Fore.WHITE)
            print("Test results: ")
            for key in self._classifierEvaluation[classifier_name]['test_results']:
                print(Fore.GREEN + key, self._classifierEvaluation[classifier_name]['test_results'][key])
            print()