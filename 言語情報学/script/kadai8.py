# usage: python kadai.py PATH
import sys

args = sys.argv
path = args[1]
with open(path) as f:
    l = f.read()
l_list = l.split('\n')[:-1]

labels = []
vecs = []
for line in l_list:
    line_ = line.split(' ')
    labels.append(line_[0])
    vecs.append(tuple([int(i) for i in line_[-1].split(',')]))

print(labels)
print(vecs)