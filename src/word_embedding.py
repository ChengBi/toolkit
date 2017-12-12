import numpy as np
import re
import pickle

class corpusClean():

    def __init__(self, filename):
        self.filename = filename
        self.num_re = re.compile('[0-9一二三四五六七八九十]+')
        self.char_re = re.compile('[a-zA-Z\-\,\.\。\，]+')

    def clean(self):
        lines = open(self.filename).readlines()
        cleaned = open('../data/cleaned_corpus.txt', 'w', encoding='utf-8')
        for line in lines:
            sentence = line.split()[-1]
            removed_char = re.sub(self.char_re, ' <CHAR> ', sentence)
            removed_num = re.sub(self.num_re, ' <NUMBER> ', removed_char)
            words = removed_num.split()
            saved_words = ['<BEGIN>']
            for word in words:
                if len(word) == 1 or word.find('<') != -1:
                    saved_words.append(word)
                else:
                    for w in word:
                        saved_words.append(w)
            saved_words.append('<END>')
            saved_sentence = ' '.join(saved_words)
            cleaned.write(saved_sentence+'\n')
        cleaned.close()
        
class word2vector():

    def __init__(self, filename, n_gram, read_flag):
        self.filename = filename
        self.n_gram = n_gram
        self.lines = open(self.filename, encoding='utf-8').readlines()
        #self.lines = self.lines[:5000]
        if read_flag == False:
            self.encode()
        else:
            self.word_encoding = pickle.load(open('../data/word_encoding.npz', 'rb'))
            self.word_map = pickle.load(open('../data/word_map.npz', 'rb'))
            self.U = pickle.load(open('../data/U.npz', 'rb'))
            self.S = pickle.load(open('../data/S.npz', 'rb'))
            self.V = pickle.load(open('../data/V.npz', 'rb'))
            
    def encode(self):
#         if use_svd == True:
#             assert k!=None
#         else:
#             self.k = k
        word_set = set()
        for line in self.lines:
            words = line.split()
            for word in words:
                word_set.add(word)
        word_set.add('UNKNOWN')
        word_set = list(word_set)
        self.word_map = dict()
        for i in range(len(word_set)):
            self.word_map[word_set[i]] = i
        self.word_encoding = np.identity(len(self.word_map))
        # pickle.dump(self.word_encoding, open('../data/word_encoding.npz', 'wb'))
        # self.word_encoding = np.zeros((len(self.word_map), len(self.word_map)))
        for line in self.lines:
            words = line.split()
            for i in range(len(words)-self.n_gram):
                neighbours = words[i:i+self.n_gram-1]+words[i+self.n_gram:i+2*self.n_gram-1]
                for neighbour in neighbours:
                    self.word_encoding[self.getIndex(words[i+self.n_gram-1])][self.getIndex(neighbour)] += 1
        for i in range(self.word_encoding.shape[0]):
            self.word_encoding[i] = self.word_encoding[i] / np.sum(self.word_encoding[i])
#         print(self.word_encoding.shape)

#         if use_svd == True:
#             self.word_encoding_svd = self.SVD(self.word_encoding, 2000)
#         else:
#         self.word_encoding_svd = self.word_encoding

        pickle.dump(self.word_encoding, open('../data/word_encoding.npz', 'wb'))
        pickle.dump(self.word_map, open('../data/word_map.npz', 'wb'))

    def decode(self, words):
        result = np.zeros(len(self.word_map))
        for word in words:
            index = self.getIndex(word)
            result[index] += 1
        result = result/np.sum(result)
#         print(result.shape)
#         print(self.S.shape)
#         result = np.dot(result, self.S)
        return result

    def decodeSentence(self, sentence):
        words = sentence.split()
        single_vector = np.zeros((len(words)-2, len(self.word_map)))
        ngram_vector = np.zeros((len(words)-2, len(self.word_map)))
        for i in range(len(words)-self.n_gram):
            single_vector[i] = self.decode(words[i+self.n_gram-1])
            neighbour = words[i:i+self.n_gram-1]+words[i+self.n_gram:i+2*self.n_gram-1]
            ngram_vector[i] = self.decode(neighbour)
        return single_vector, ngram_vector

    def get(self):
        self.data = dict()
#         self.data['inputs'] = [None]*len(self.lines)
#         self.data['targets'] = [None]*len(self.lines)
        
        self.data['inputs'] = dict()
        self.data['targets'] = dict()
        self.data['batch_size'] = dict()

        for line, i in zip(self.lines, range(len(self.lines))):
            single, ngram = self.decodeSentence(line)
            size = single.shape[0]
            if size not in self.data['batch_size'].keys():
                self.data['inputs'][size] = []
                self.data['targets'][size] = []
                self.data['batch_size'][size] = 0
            self.data['inputs'][size].append(single)
            self.data['targets'][size].append(ngram)
            self.data['batch_size'][size] += 1
        
#         for key in self.data['batch_size'].keys():
#             print(self.data['batch_size'][key])
#             self.data['inputs'][key] = [None]*self.data['batch_size'][key]
#             self.data['targets'][key] = [None]*self.data['batch_size'][key]
        
#             print(key)
            
        
#             self.data['inputs'][i] = single
#             self.data['targets'][i] = ngram
#         self.data['inputs'] = np.array(self.data['inputs'])
#         self.data['targets'] = np.array(self.data['targets'])
        
#         batches = dict()
#         batches['inputs'] = dict()
#         batches['targets'] = dict()
#         batches['keys'] = dict()
#         for i, j in zip(self.data['inputs'], self.data['targets']):
#             key = i.shape[0]
#             if key not in batches.keys():
#                 batches['inputs'][key] = []
#                 batches['targets'][key] = []
#                 batches['keys'][key] = 0
#             batches['inputs'][key].append(i)
#             batches['targets'][key].append(j)
#             batches['keys'][key] += 1
            
#         for key in batches['keys'].keys():
#             batches['inputs'][key] = np.array(batches['inputs'][key])
#             batches['targets'][key] = np.array(batches['targets'][key])
#             batches['keys'][key] = np.array(batches['keys'][key])
        
#         self.batches = batches
        pickle.dump(self.data, open('../data/data.npz', 'wb'))
#         pickle.dump(self.batches, open('../data/batches_data.npz', 'wb'))
#         print(self.data['inputs'].shape)
#         print(self.data['targets'].shape)
        return self.data#, self.batches

    def SVD(self, inputs, k):
        self.U, self.S, self.V = np.linalg.svd(inputs)
        self.U = self.U[:, :k]
        self.S = np.diag(self.S)[:k, :k]
        self.V = self.V[:k, :]
        pickle.dump(self.U, open('../data/U.npz', 'wb'))
        pickle.dump(self.S, open('../data/S.npz', 'wb'))
        pickle.dump(self.V, open('../data/V.npz', 'wb'))
        return self.U.dot(self.S)

    def getIndex(self, word):
        if word in self.word_map.keys():
            return self.word_map[word]
        else:
            return self.word_map['UNKNOWN']
        
lines = open('../data/cleaned_corpus.txt', encoding='utf-8').readlines()
lines_set = set()
for line in lines:
    lines_set.add(line)
print(len(lines_set))
writer = open('../data/trimmed_cleaned_corpus.txt', 'w', encoding='utf-8')
for line in list(lines_set):
    writer.write(line)
writer.close()

w2v = word2vector('../data/trimmed_cleaned_corpus.txt', 2, True)
data = w2v.get()

