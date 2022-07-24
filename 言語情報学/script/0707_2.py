import sys
def computeTF(word_dict, bow):
    tf_dict = {}
    bow_cnt = len(bow)
    for word, cnt in word_dict.items():
        tf_dict[word] = cnt / float(bow_cnt)
    return tf_dict

args = sys.argv
path = args[1]
with open(path) as f:
    doc = f.read()
bag_of_words = ([word.lower() for word in doc.split()])
unique = set(bag_of_words)
num_words = dict(zip(unique, [0]*len(unique)))
for word in bag_of_words:
    num_words[word] += 1

tf = computeTF(num_words, bag_of_words)
top5 = sorted([[value, key] for key, value in tf.items()], reverse=True)[:5]
for i, value_key in enumerate(top5):
    value, key = value_key
    print(f'{i}: {key} {value}')

