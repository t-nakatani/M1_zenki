#usage python 0714_2.py FILE_PATH
import sys
import glob
import math
import pandas as pd
def computeTF(word_dict, bow):
    tf_dict = {}
    bow_cnt = len(bow)
    for word, cnt in word_dict.items():
        tf_dict[word] = cnt / float(bow_cnt)
    return tf_dict

def computeIDF(doc_dict):
    N = len(doc_dict)
    idf_dict = dict(zip(doc_dict[0].keys(), [0]*len(doc_dict[0])) )
    for doc in doc_dict:
        for word, val in doc.items():
            if val > 0:
                idf_dict[word] += 1
    
    for word, val in idf_dict.items():
        idf_dict[word] = math.log(N / float(val))
    return idf_dict

def computeTFIDF(tf_bow, idfs):
    tfidf = {}
    for word, val in tf_bow.items():
        tfidf[word] = val * idfs[word]
    return tfidf

def doc2TFIDF(doc_list, N=5):
    # Bag of Words
    # bow_list = [doc.split(' ') for doc in doc_list]
    bow_list = [[w for w in doc.split(' ') if w != '' and w != '\n'] for doc in doc_list]
    unique = set(sum(bow_list, []))
    # print(unique)
    
    list_num_words = [dict(zip(unique, [0]*len(unique))) for i in range(N)]
    for i in range(5):
        for word in bow_list[i]:
            list_num_words[i][word] += 1

    list_tf = [computeTF(num_words, bow) for num_words, bow in zip(list_num_words, bow_list)]

    doc_dict = [num_words for num_words in list_num_words]
    idfs = computeIDF(doc_dict)

    tfidf_list = [computeTFIDF(tf, idfs) for tf in list_tf]
    df = pd.DataFrame(tfidf_list)

    return df, idfs

args = sys.argv
path = args[1] if len(args) == 2 else '../LI22/20220630/data*.txt'

path_list = [fname for fname in glob.glob(path) if '_' not in fname.split('/')[-1]]
doc_list = []
for path in path_list:
    with open(path) as f:
        doc = f.read()
        doc_list.append(doc.lower())
df, idfs = doc2TFIDF(doc_list, N=5)
top10s = []
for index, row in df.iterrows():
    top10s.extend(sorted([[col, i, index]for i, col in enumerate(row)], reverse=True)[:10])
for score, col, row in sorted(top10s, reverse=True)[:10]:
    print(score, df.columns[col], f'doc{row}')

# print('top10-idf')
# for score, col, row in sorted(top10s, reverse=True)[:10]:
#     print(idfs[df.columns[col]], df.columns[col])