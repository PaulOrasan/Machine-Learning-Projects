from os.path import isfile

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score

lookup = {0: 'T-shirt',
          1: 'Trouser',
          2: 'Pullover',
          3: 'Dress',
          4: 'Coat',
          5: 'Sandal',
          6: 'Shirt',
          7: 'Sneaker',
          8: 'Bag',
          9: 'Ankle boot'}

def display_image(features, actual_label, predicted_label):
    plt.figure()
    plt.text(-10, -2, 'Actual label: ' + lookup[actual_label])
    plt.text(20, -2, 'Predicted label: ' + lookup[predicted_label])
    plt.imshow(features.reshape(28,28))
    plt.show()


train_df = pd.read_csv('../../datasets/fashion_train.csv')
print(train_df.head())
print(train_df.shape)
x_train = train_df.drop(labels=['y'], axis=1)
y_train = train_df['y']

print(x_train.head())
print(y_train.head())

test_df = pd.read_csv('../../datasets/fashion_test.csv')
print(test_df.head())
print(test_df.shape)
x_test = test_df.drop(labels=['y'], axis=1)
y_test = test_df['y']

print(x_test.head())
print(y_test.head())
model = None
if isfile('save.sav'):
    model = joblib.load('save.sav')
else:
    model = LogisticRegression(solver='sag', multi_class='auto', max_iter=10000).fit(x_train, y_train)
    joblib.dump(model, 'save.sav')

y_pred = model.predict(x_test)
print('Accuracy: ', accuracy_score(y_test, y_pred, normalize=True))
print('Precision: ', precision_score(y_test, y_pred, average='weighted'))
print('Recall: ', recall_score(y_test, y_pred, average='weighted'))

option = 0
while option != -1:
    option = int(input("Please ask for an image: "))
    display_image(x_test.loc[option].values, y_test[option], y_pred[option])