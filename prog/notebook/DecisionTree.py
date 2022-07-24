import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

X, y = make_blobs(random_state=1, n_features=2, centers=3, cluster_std=0.6, n_samples=300)
X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, random_state=1)
TreeModel = DecisionTreeClassifier(max_depth=None, random_state=1)
TreeModel.fit(X_tr, y_tr)

pred = TreeModel.predict(X_ts)
score = accuracy_score(y_ts, pred)
# print('Accuracy: {}%'.format(score*100))

df = pd.DataFrame(X_ts)
plt.figure(figsize=(15, 12))
plot_tree(TreeModel, fontsize=20, filled=True, feature_names=['df[0]', 'df[1]'], class_names=['0', '1', '2'])