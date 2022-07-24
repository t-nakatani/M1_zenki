import sys
import re
def remove_noise(word_line):
    result = []
    words = word_line.split(' ')
    for word in words:
        if re.match('<.*>', word):
            continue
        elif re.match('@.*', word):
            continue
        elif re.match('\'s', word):
            continue
        elif re.match('\d+', word):
            continue
        elif re.match('[a-zA-Z]+', word):
            result.append(word)
    return ' '.join(result)

args = sys.argv
path = args[1]
print(path)
result = []
with open(path) as f:
    for i, line in enumerate(f):
        if line.isspace():
            # print('space line is detected !\n')
            continue
        line=line.strip() 
        result.append(remove_noise(line))
print(path.replace('data_org', 'result'))
with open(path.replace('data_org', 'result'),"w") as f:
    for line in result:
        f.write(line+"\n")
# print(f'num lines in document == {i}')