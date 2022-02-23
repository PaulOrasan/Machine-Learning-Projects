import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


class Dataset:

    def __init__(self, file_path, columns, preprocesser=None):
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_columns', 500)
        self._encoders = {}
        self._dataframe = pd.read_csv(file_path, names=columns)
        if preprocesser != None:
            preprocesser(self)

    def dropColumn(self, column):
        self._dataframe = self._dataframe.drop(column)

    def encodeColumnToNumericFormat(self, feature: str):
        encoder = LabelEncoder()
        self._dataframe[feature] = encoder.fit_transform(self._dataframe[feature].astype('str'))
        self._encoders[feature] = encoder

    def encodeMultipleColumnsToNumerifcFormat(self, features: list):
        for feature in features:
            self.encodeColumnToNumericFormat(feature)

    def splitXY(self, features, label):
        return (self._dataframe.drop(labels=[label], axis=1), self._dataframe[label])

    def showDescription(self):
        print(self._dataframe.describe(include='all'))

    def showHead(self):
        print(self._dataframe.head())

    def showDataTypes(self):
        print(self._dataframe.dtypes)

    def categoricalPlot(self, feature, label='income'):
        df = self._dataframe[[feature, label]]
        df[feature] = self._encoders[feature].inverse_transform(df[feature])
        df = df.groupby(feature)[label].value_counts(normalize=True)
        df = df.rename('percent').reset_index()
        print(self._dataframe[[feature, label]].sample(frac=0.1).shape)
        sns.catplot(x=feature, hue=label, y='percent', kind='bar', data=df)

    def showPlots(self):
        plt.show()