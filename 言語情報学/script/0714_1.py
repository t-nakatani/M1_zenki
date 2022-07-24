#usage python 0714_1.py FILE_PATH
import sys
import re
import pandas as pd
from sklearn.model_selection import train_test_split

args = sys.argv
path = args[1] if len(args) == 2 else '../LI22/20220707/dataset.txt'


with open(path) as f:
    l = f.readlines()
    dataset = pd.DataFrame([list(map(int, re.findall('\d+', line))) for line in l])
dataset.columns = ['label'] + list(range(10))
t = dataset.label
v = dataset.iloc[:, 1:]


train_index, test_index = train_test_split(range(len(t)), test_size=0.3, random_state=0)
train_index = sorted(train_index)
test_index = sorted(test_index)
data_train = v.iloc[train_index]
label_train = t.iloc[train_index]
data_test = v.iloc[test_index]
label_test = t.iloc[test_index]

print('shape of (data_tr data_ts label_tr label_ts = ', data_train.shape, data_test.shape, label_train.shape, label_test.shape)