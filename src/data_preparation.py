import gensim
import numpy as np
import pickle

label_map = pickle.load(open('gensim_word2vec_label_map.npz', 'rb'))
lines = open('train.txt').readlines()
corpus = [line.split()[1:] for line in lines]

vector_size = 200
ngram = 3
min_count = 2
workers = 4

models = gensim.models.Word2Vec(corpus, size=vector_size, window=ngram, min_count=min_count, workers=workers)
models.save('word2vec_anouymous.model')

class dataProvider():
    
    def __init__(self,filename, models, label_map):
        
        self.filename = filename
        self.models = models
        self.size = self.models.vector_size
        self.label_map = label_map
        self.data = dict()
        self.data['inputs'] = dict()
        self.data['targets'] = dict()
        self.lines = open(filename).readlines()
    
    def extract(self):
        
        for line in self.lines:
            values = line.split()
            words = values[1:]
            tag = values[0]
            key = len(words)
            inputs = np.zeros((len(words), self.size))
            targets = np.zeros(len(self.label_map))
            for word, i in zip(words, range(len(words))):
                try:
                    inputs[i] = self.models.wv[word]
                except:
                    continue
            targets[int(tag)] = 1.0
            if key not in self.data['inputs'].keys():
                self.data['inputs'][key] = []
                self.data['targets'][key] = []
            self.data['inputs'][key].append(inputs)
            self.data['targets'][key].append(targets)
            
models = gensim.models.Word2Vec.load('word2vec_anouymous.model')

train_provider = dataProvider('train.txt', models, label_map)
train_provider.extract()
#pickle.dump(train_provider.data, open('../word2vec/anouymous_train_data.npz', 'wb'))

valid_provider = dataProvider('valid.txt', models, label_map)
valid_provider.extract()
#pickle.dump(valid_provider.data, open('../word2vec/anouymous_valid_data.npz', 'wb'))

def batch(data, batch_size):
    
    keys = data['inputs'].keys()
    results = dict()
    results['inputs'] = dict()
    results['targets'] = dict()
    
    for key in keys:
        count = int(len(data['inputs'][key])/batch_size)+1
        for i in range(count):
            new_key = str(key)+'_'+str(i)
            if i+1 != count:
                results['inputs'][new_key] = np.array(data['inputs'][key][i*batch_size:(i+1)*batch_size])
                results['targets'][new_key] = np.array(data['targets'][key][i*batch_size:(i+1)*batch_size])
            else:
                results['inputs'][new_key] = np.array(data['inputs'][key][i*batch_size:])
                results['targets'][new_key] = np.array(data['targets'][key][i*batch_size:])
    return results

#train_batch = batch(train_provider.data, 50)
#valid_batch = batch(valid_provider.data, 50)
#pickle.dump(train_batch, open('anouymous_train_data_50.npz', 'wb'))
#pickle.dump(valid_batch, open('anouymous_valid_data_50.npz', 'wb'))

train_batch = batch(train_provider.data, 200)
valid_batch = batch(valid_provider.data, 200)
pickle.dump(train_batch, open('anouymous_train_data_200.npz', 'wb'))
pickle.dump(valid_batch, open('anouymous_valid_data_200.npz', 'wb'))