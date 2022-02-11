import pandas as pd

from ClassifierManager import ClassifierManager

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import GridSearchCV, train_test_split

ionosphere_df = pd.read_csv('../../datasets/ionosphere.data', names=[str(i) for i in range(1, 34)] + ['status'])
print("Head: ")
print(ionosphere_df.head())

print("Description: ")
print(ionosphere_df.describe())

print("Count of missing values:")
print(ionosphere_df[ionosphere_df.isnull().any(axis=1)].count())

print("Shape before dropping empty values: ", end="")
print(ionosphere_df.shape)

print("Shape after dropping empty values: ", end="")
ionosphere_df = ionosphere_df.dropna()
print(ionosphere_df.shape)

print("Converting status column from text to numbers: ")
ionosphere_df['status'] = ionosphere_df['status'].apply(lambda x: 1 if x == 'g' else 0)
print(ionosphere_df.head())

X = ionosphere_df.drop(labels=['status'], axis=1)
Y = ionosphere_df['status']

def hyperparameterTuning(model, parameters, X, Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)
    searcher = GridSearchCV(model, parameters, cv=3, return_train_score=True)
    searcher.fit(x_train, y_train)
    return searcher

classifierManager = ClassifierManager(X, Y)

classifierManager.addClassifier('logistic_regression no hyperparameters', LogisticRegression(solver='liblinear'))
classifierManager.addClassifier('Linear SVC with 1000 max iterations and C=1', LinearSVC(max_iter=1000, C=1, dual=False))
classifierManager.addClassifier('KNN with K=10', KNeighborsClassifier(n_neighbors=10))
classifierManager.addClassifier('Decision Tree with no max depth and features (all leafs)', DecisionTreeClassifier(max_depth=None, max_features=None))
classifierManager.addClassifier('Naive Bayes with no hyperparameters', GaussianNB())

parametersLogisticRegression = {'penalty': ['l1', 'l2'], 'C':[0.1, 0.2, 0.5, 1, 2, 5]}
parametersKNN = {'n_neighbors':[3, 5, 8, 10]}
parametersDecisionTree = {'max_depth': [2, 4, 5, 7, 10]}

bestModel = hyperparameterTuning(LogisticRegression(solver='liblinear'), parametersLogisticRegression, X, Y)
print('Best value for penalty and C are: ', bestModel.best_params_)
print()
classifierManager.addClassifier('Logistic Regression with tuned hyperparameter',
                                LogisticRegression(penalty=bestModel.best_params_['penalty'], C=bestModel.best_params_['C'], solver='liblinear'))

bestModel = hyperparameterTuning(KNeighborsClassifier(), parametersKNN, X, Y)
print('Best value for K is: ', bestModel.best_params_)
print()
classifierManager.addClassifier('KNN with tuned hyperparameter', KNeighborsClassifier(n_neighbors=bestModel.best_params_['n_neighbors']))

bestModel = hyperparameterTuning(DecisionTreeClassifier(), parametersDecisionTree, X, Y)
print('Best value for max_depth is: ', bestModel.best_params_)
print()
classifierManager.addClassifier('Decision tree with tuned hyperparameter',
                                DecisionTreeClassifier(max_depth=bestModel.best_params_['max_depth'], max_features=None))

classifierManager.showEvaluations()