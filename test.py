import numpy as np
from dl85 import DL85Classifier
from sklearn.model_selection import train_test_split
import pandas as pd

"""
dataset = np.genfromtxt("datasets/anneal.txt", delimiter=' ')
X, y = dataset[:, 1:], dataset[:, 0]

"""
col1 = np.array([0.0] * 10 + [1.0] * 0 + [0.0] * 0 + [1.0] * 10).reshape(-1, 1)
col2 = np.array([0.0] * 5 + [1.0] * 5 + [0.0] * 5 + [1.0] * 5).reshape(-1, 1)
X = np.array([0.0] * 10 + [1.0] * 0 + [0.0] * 0 + [1.0] * 10)
X = np.concatenate((col1, col2), axis=1)

y = np.array([0.0] * 8 + [1.0] * 2 + [0.0] * 2 + [1.0] * 8)
sensitive = np.array([0.0] * 6 + [1.0] * 5 + [0.0] * 7 + [1.0] * 2)
# X_train,y_train = X, y

# DATAFRAME
df = pd.DataFrame(X)
df['Class'] = y.tolist()
df['Sensitive'] = sensitive.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


def train_test_split(random_state):
    global X_train, y_train, sensitive_train
    global X_test, y_test, sensitive_test

    X = df.loc[:, ~df.columns.isin(['Class', 'Sensitive'])]
    y = df['Class']
    sensitive = df['Sensitive']

    index_train = list(df.sample(frac=1, random_state=random_state).index)
    index_test = list(df.drop(index=index_train).index)

    X_train = X.drop(index=index_test).to_numpy()
    y_train = y.drop(index=index_test).to_numpy()
    sensitive_train = sensitive.drop(index=index_test).to_numpy()

    X_test = X.drop(index=index_train).to_numpy()
    y_test = list(y.drop(index=index_train).to_numpy())
    sensitive_test = sensitive.drop(index=index_train).to_numpy()


train_test_split(1)


def error(tids):
    classes, supports = np.unique(y_train.take(list(tids)), return_counts=True)
    maxindex = np.argmax(supports)
    return sum(supports) - supports[maxindex], classes[maxindex]


clf = DL85Classifier(max_depth=5, error_function=lambda tids: error(list(tids)), min_sup=0, time_limit=600)
clf.fit(X_train, list(y_train))
