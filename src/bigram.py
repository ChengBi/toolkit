import numpy as np

lines = open('../data/corpus.txt').readlines()
word_set = set()

for line in lines:
    words = line.split()
    for word in words:
        word_set.add(word)
word_set = list(word_set)
word_id = np.arange(len(word_set))
# print(len(word_set))
n_word = len(word_set)
word_mat = np.zeros((n_word, n_word))
word_set.sort()
print('------------------')
print(word_set)
print('------------------')
for line in lines:
    words = line.split()
    if len(words) == 1:
        # word_mat[word_set.index(words[0]), -1] += 1
        continue
    else:
        for i, j in zip(words[:-1], words[1:]):
            word_mat[word_set.index(i), word_set.index(j)] += 1

print(word_mat)
print('------------------')
graph = dict()
for i in range(n_word):
    for j in range(n_word):
        if word_mat[i, j] != 0.0:
            graph[(i, j)] = word_mat[i, j]
print(graph)
head_set = set()
tail_set = set()
for key, value in graph.items():
    head_set.add(key[0])
    tail_set.add(key[1])

start_node_id = []
end_node_id = []
for i in word_id:
    if i in head_set and i not in tail_set:
        start_node_id.append(i)
    if i in tail_set and i not in head_set:
        end_node_id.append(i)

print(start_node_id)
print(end_node_id)
