import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from RegressionManager import RegressionManager
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge

wine_df = pd.read_csv('../../datasets/winequality-white.csv', sep=';')
print(wine_df.head())
print(wine_df.dtypes)
print(wine_df.describe())
print(wine_df.shape)
label = 'quality'
def plotColumns(dataframe, feature, label):
    plt.figure()
    plt.scatter(dataframe[feature], dataframe[label])
    plt.xlabel(feature)
    plt.ylabel(label)
    plt.show()
'''
plotColumns(wine_df, 'fixed acidity', label)
plotColumns(wine_df, 'volatile acidity', label)
plotColumns(wine_df, 'citric acid', label)
plotColumns(wine_df, 'residual sugar', label)
plotColumns(wine_df, 'chlorides', label)
plotColumns(wine_df, 'free sulfur dioxide', label)
plotColumns(wine_df, 'total sulfur dioxide', label)
plotColumns(wine_df, 'density', label)
plotColumns(wine_df, 'pH', label)
plotColumns(wine_df, 'sulphates', label)
plotColumns(wine_df, 'alcohol', label)'''
'''
plt.scatter(wine_df['alcohol'], wine_df['quality'])
plt.xlabel('alcohol')
plt.ylabel('quality')
plt.show()

'''

print(wine_df.corr())
sbn.heatmap(wine_df.corr(), annot=True)
plt.show()

modelManager = RegressionManager(wine_df)
modelManager.addModel('linear regression', LinearRegression(), ['alcohol'], label)
modelManager.addModel('linear regression', LinearRegression(), ['fixed acidity', 'volatile acidity', 'citric acid'], label)
modelManager.addModel('linear regression', LinearRegression(), ['residual sugar'], label)
modelManager.addModel('linear regression', LinearRegression(), ['chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'sulphates'], label)
modelManager.addModel('linear regression', LinearRegression(), ['density', 'pH', 'alcohol'], label)
modelManager.addModel('linear regression', LinearRegression(), ['fixed acidity', 'volatile acidity', 'citric acid', 'chlorides',
                                'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'],
                      label)

modelManager = RegressionManager(wine_df)
modelManager.addModel('linear regression', LinearRegression(), ['alcohol', 'density', 'total sulfur dioxide',
                                                                              'chlorides', 'volatile acidity'], label)
modelManager.addModel('lasso with alpha 0.5', Lasso(alpha=0.5), ['alcohol', 'density', 'total sulfur dioxide',
                                                                              'chlorides', 'volatile acidity'], label)
modelManager.addModel('ridge with alpha 0.5', Ridge(alpha=0.5), ['alcohol', 'density', 'total sulfur dioxide',
                                                                              'chlorides', 'volatile acidity'], label)
modelManager.addModel('KNR with k=10', KNeighborsRegressor(n_neighbors=10), ['alcohol', 'density', 'total sulfur dioxide',
                                                                              'chlorides', 'volatile acidity'], label)
modelManager.addModel('SGD', SGDRegressor(), ['alcohol', 'density', 'total sulfur dioxide',
                                                                              'chlorides', 'volatile acidity'], label)
modelManager.showEvaluations()
if input("Do you want to see plots? y/n") == 'y':
    plt.show()
