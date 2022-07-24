from sklearn.datasets import load_iris, load_digits
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.metrics import accuracy_score

data_iris = load_iris()
X = data_iris.data
y = data_iris.target
X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3, random_state=0)

param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
print('グリッドサーチの範囲\n', param_grid, '\n')
clf = GridSearchCV(svm.SVC(), param_grid, n_jobs = -1)
clf.fit(X_tr, y_tr)
print('最適なパラメータ\nC_opt: {}, gamma_opt: {}\n'.format(clf.best_params_['C'], clf.best_params_['gamma']))
y_ts_pred = clf.predict(X_ts)
print('best accuracy\n', accuracy_score(y_ts_pred, y_ts))