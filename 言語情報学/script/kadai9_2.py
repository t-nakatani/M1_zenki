# usage: python kadai9_1.py PATH
# ex.) python kadai9_1.py ../LI22/20220623/data0623.csv
import sys
import pandas as pd
import numpy as np

args = sys.argv
path_csv = args[1]

# path_csv = '../LI22/20220623/data0623.csv'
df = pd.read_csv(path_csv, header=None)
df.iloc[0,:] = sorted(df.iloc[0,:])
df.iloc[1,:] = sorted(df.iloc[1,:], reverse=True)
df.to_csv(path_csv.replace('data0623', 'result'), index=False, header=False)