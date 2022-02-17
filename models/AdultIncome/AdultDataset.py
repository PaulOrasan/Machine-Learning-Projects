import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


class AdultDataset:

    def __init__(self, file_path):
        self._features = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'martial-status', 'occupation',
                          'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                          'native-country',
                          'income']
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_columns', 500)
        self._encoders = {}
        self._dataframe = pd.read_csv(file_path, names=self._features)
        self._dataframe = self._dropIrrelevantColumns(self._dataframe)
        self._encodeMultipleColumnsToNumerifcFormat(self._dataframe, ['workclass', 'education', 'martial-status', 'occupation',
                                                      'relationship', 'race', 'sex', 'native-country', 'income'])

    def _dropIrrelevantColumns(self, dataframe):
        return dataframe

    def _encodeColumnToNumericFormat(self, dataframe, feature: str):
        encoder = LabelEncoder()
        dataframe[feature] = encoder.fit_transform(dataframe[feature].astype('str'))
        self._encoders[feature] = encoder
        print(encoder.__dict__)

    def _encodeMultipleColumnsToNumerifcFormat(self, dataframe, features: list):
        for feature in features:
            self._encodeColumnToNumericFormat(self._dataframe, feature)

    def showDescription(self):
        print(self._dataframe.describe(include='all'))

    def showHead(self):
        print(self._dataframe.head())

    def showDataTypes(self):
        print(self._dataframe.dtypes)

    def scatterPlot(self, feature, label='income'):
        plt.figure()
        plt.scatter(self._dataframe[feature], self._dataframe[label])
        plt.xlabel(feature)
        plt.ylabel(label)
        plt.show()

    def categoricalPlot(self, feature, label='income'):
        df = self._dataframe[[feature, label]].sample(frac=0.1)
        df[feature] = self._encoders[feature].inverse_transform(df[feature])
        df = df.groupby(feature)[label].value_counts(normalize=True)
        df = df.rename('percent').reset_index()
        print(self._dataframe[[feature, label]].sample(frac=0.1).shape)
        sns.catplot(x=feature, hue=label, y='percent', kind='bar', data=df)

    def showPlots(self):
        plt.show()