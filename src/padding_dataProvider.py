import gensim
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import re
from tqdm import tqdm, tqdm_notebook

class paddingDataProvider():
    
    def __init__(self,filename, models, label_map, max_padding = 30):
        
        self.filename = filename
        self.models = models
        self.size = self.models.vector_size
        self.label_map = label_map
        self.lines = open(filename).readlines()
        self.n = len(self.lines)
        self.max_padding = max_padding
        self.data = dict()
        self.data['inputs'] = np.zeros((self.n, max_padding, self.size))
        self.data['targets'] = np.zeros((self.n, len(self.label_map)))
#         print(self.data['inputs'].shape)
#         print(self.data['targets'].shape)
        
    
    def extract(self):
#         pass
        for line, i in zip(self.lines, range(self.n)):
            values = line.split()
            words = values[1:]
            tag = values[0]
#             print(words, tag)
#             break
            inputs_temp = np.zeros((self.max_padding, self.size))
            for word, j in zip(words, range(len(words))):
#                 print(word)
                try:
                    if j >= self.max_padding:
                        break
                    inputs_temp[j] = self.models.wv[str(word)]
                except:
                    continue
#                 inputs_temp = np.zeros(self.max_padding)
#                 inputs_temp[:len(words)] = words
#             print(inputs_temp)
#                 break
#             break
            self.data['inputs'][i] = inputs_temp
            self.data['targets'][i][int(tag)] = 1.0
#             print(self.data['inputs'][i])
#             print(self.data['targets'][i])
#             break
#             key = len(words)
#             inputs = np.zeros((len(words), self.size))
#             targets = np.zeros(len(self.label_map))
#             for word, i in zip(words, range(len(words))):
#                 try:
#                     inputs[i] = self.models.wv[word]
#                 except:
#                     continue
#             targets[int(tag)] = 1.0
#             if key not in self.data['inputs'].keys():
#                 self.data['inputs'][key] = []
#                 self.data['targets'][key] = []
#             self.data['inputs'][key].append(inputs)
#             self.data['targets'][key].append(targets)


models = gensim.models.Word2Vec.load('../word2vec/word2vec_anouymous.model')
label_map = pickle.load(open('../word2vec/gensim_word2vec_label_map.npz', 'rb'))
train_provider = paddingDataProvider('../word2vec/train.txt', models, label_map, max_padding=20)
train_provider.extract()
pickle.dump(train_provider.data, open('../word2vec/anouymous_padded_train_data.npz', 'wb'))

valid_provider = paddingDataProvider('../word2vec/valid.txt', models, label_map, max_padding=20)
valid_provider.extract()
pickle.dump(valid_provider.data, open('../word2vec/anouymous_padded_valid_data.npz', 'wb'))