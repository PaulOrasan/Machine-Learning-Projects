from Classifier import Classifier
from Dataset import Dataset
from Manager import ClassifierManager
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

def preprocessAdultDataset(dataframe: Dataset):
    dataframe.encodeMultipleColumnsToNumerifcFormat(text_features)

train_file_path = '../../datasets/adult.data'
test_file_path = '../../datasets/adult.test'
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'martial-status', 'occupation',
                          'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                          'native-country', 'income']
features = columns[:].remove('income')
label = 'income'

text_features = ['workclass', 'education', 'martial-status', 'occupation', 'relationship', 'race', 'sex',
                 'native-country', 'income']

train_dataset = Dataset(train_file_path, columns, preprocessAdultDataset)

train_dataset.showDataTypes()
train_dataset.showDescription()
'''
ANALYSIS
train_dataset.categoricalPlot('race')
train_dataset.categoricalPlot('sex')
train_dataset.categoricalPlot('education')
train_dataset.categoricalPlot('native-country')
train_dataset.showDataTypes()
train_dataset.showPlots()
'''
test_dataset = Dataset(test_file_path, columns, preprocessAdultDataset)
test_dataset.showDescription()
test_dataset.showDataTypes()

manager = ClassifierManager()

manager.addModel(Classifier('Logistic Regression with default hyperparameters', LogisticRegression(solver='sag'),
                            train_dataset, features, label))
manager.addModel(Classifier('Linear SVC with default hyperparameters and 1000 iterations', SVC(max_iter=1000),
                            train_dataset, features, label))
manager.addModel(Classifier('KNN with 10 neighbours', KNeighborsClassifier(n_neighbors=10),
                            train_dataset, features, label))
manager.addModel(Classifier('Decision Tree with no max depth and features (all leafs)', DecisionTreeClassifier(max_depth=None, max_features=None),
                            train_dataset, features, label))
manager.addModel(Classifier('Naive Bayes with default hyperparameters', GaussianNB(),
                            train_dataset, features, label))
manager.addModel(Classifier('Multi-layer pereptron with default hyperparameters', MLPClassifier(), train_dataset, features, label))
manager.addModel(Classifier('Multi-layer pereptron with 3 layers', MLPClassifier((10,5,7)), train_dataset, features, label))
manager.testModels(test_dataset)
manager.showEvaluations()