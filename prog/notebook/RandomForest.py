import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data=load_wine()
X, y = data.data, data.target
X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3, random_state=0)
TreeModel = RandomForestClassifier()
TreeModel.fit(X_tr, y_tr)

pred = TreeModel.predict(X_ts)
score = accuracy_score(y_ts, pred)
print('Accuracy: {}%'.format(score*100))

for name, val in zip(data['feature_names'], TreeModel.feature_importances_):
    print('{} :\n    {}'.format(name, val))

df_wine = pd.DataFrame([[name, val]for name, val in zip(data['feature_names'], TreeModel.feature_importances_)])
df_wine = df_wine.sort_values(1)
plt.figure(figsize=(15, 12))
plt.title('feature importances', fontsize=18)
plt.xlabel('importance scores', fontsize=15)
plt.ylabel('features', fontsize=15)
plt.barh(df_wine[0], df_wine[1])
plt.show()