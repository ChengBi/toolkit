import gensim
import numpy as np
import pickle

class corpusClean():

    def __init__(self):
        self.num_re = re.compile('[0-9一二三四五六七八九十]+')
        self.char_re = re.compile('[a-zA-Z\-\,\.\。\，]+')

    def clean(self, line):
        sentence = line
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
        return saved_words
    
    
vector_size = 200
ngram = 3
min_count = 2
workers = 4
lines = open('../data/cleaned_corpus.txt', encoding='utf-8').readlines()
sentences = [i.split() for i in lines]
# print(sentences[0])
models = gensim.models.Word2Vec(sentences, size=vector_size, window=ngram, min_count=min_count, workers=workers)
models.save('../word2vec/word2vec.model')

lines = open('../data/TrainSet-eCarX-171019.txt').readlines()
target_set = set()
for line in lines:
    target = line.split()[0]
    target_set.add(target)
target_set = list(target_set)
print(len(target_set))
models = gensim.models.Word2Vec.load('../word2vec/word2vec.model')
cleaner = corpusClean()
data = dict()
data['train_inputs'] = dict()
data['train_targets'] = dict()
data['valid_inputs'] = dict()
data['valid_targets'] = dict()
data['label_map'] = target_set
data['batched_train_inputs'] = dict()
data['batched_train_targets'] = dict()
data['batched_valid_inputs'] = dict()
data['batched_valid_targets'] = dict()

batch_size = 50

for line in lines:
    values = line.split()
    sentence = values[-1]
    tag = values[0]
    cleaned_sentence = cleaner.clean(sentence)
    inputs = np.zeros((len(cleaned_sentence), 200))
    for word, i in zip(cleaned_sentence, range(len(cleaned_sentence))):
        try:
            inputs[i] = models.wv[word]
        except:
            continue
    targets = np.zeros(len(target_set))
    targets[target_set.index(tag)] = 1.0
    key = len(inputs)
    if key not in data['train_inputs'].keys():
        data['train_inputs'][key] = []
        data['train_targets'][key] = []
    data['train_inputs'][key].append(inputs)
    data['train_targets'][key].append(targets)
    
data['train_keys'] = data['train_inputs'].keys()
print('Feature Extraction for training data finished.')


for key in data['train_keys']:
    count = int(len(data['train_inputs'][key])/batch_size)+1
    for i in range(count):
        new_key = str(key)+'_'+str(i)
        if i+1 != count:
            data['batched_train_inputs'][new_key] = np.array(data['train_inputs'][key][i*batch_size:(i+1)*batch_size])
            data['batched_train_targets'][new_key] = np.array(data['train_targets'][key][i*batch_size:(i+1)*batch_size])
        else:
            data['batched_train_inputs'][new_key] = np.array(data['train_inputs'][key][i*batch_size:])
            data['batched_train_targets'][new_key] = np.array(data['train_targets'][key][i*batch_size:])
    if count != 1:
        if (data['batched_train_inputs'][str(key)+'_1'][0] == data['train_inputs'][key][batch_size+1]).all() == True:
            print('SUCCESS!')
            
print('Batch data extraction finished.')

lines = open('../data/TestSet-eCarX-171019.txt', encoding='gbk').readlines()
for line in lines:
    values = line.split('#')
    sentence = values[0]
    tag = values[2]
    cleaned_sentence = cleaner.clean(sentence)
    inputs = np.zeros((len(cleaned_sentence), 200))
    for word, i in zip(cleaned_sentence, range(len(cleaned_sentence))):
        try:
            inputs[i] = models.wv[word]
        except:
            continue
    targets = np.zeros(len(target_set))
    targets[target_set.index(tag)] = 1.0
    key = len(inputs)
    if key not in data['valid_inputs'].keys():
        data['valid_inputs'][key] = []
        data['valid_targets'][key] = []
    data['valid_inputs'][key].append(inputs)
    data['valid_targets'][key].append(targets)
#     break
data['valid_keys'] = data['valid_inputs'].keys()
print('Feature Extraction for validation data finished.')

for key in data['valid_keys']:
    count = int(len(data['valid_inputs'][key])/batch_size)+1
    for i in range(count):
        new_key = str(key)+'_'+str(i)
        if i+1 != count:
            data['batched_valid_inputs'][new_key] = np.array(data['valid_inputs'][key][i*batch_size:(i+1)*batch_size])
            data['batched_valid_targets'][new_key] = np.array(data['valid_targets'][key][i*batch_size:(i+1)*batch_size])
        else:
            data['batched_valid_inputs'][new_key] = np.array(data['valid_inputs'][key][i*batch_size:])
            data['batched_valid_targets'][new_key] = np.array(data['valid_targets'][key][i*batch_size:])
    if count != 1:
        if (data['batched_valid_inputs'][str(key)+'_1'][0] == data['valid_inputs'][key][batch_size+1]).all() == True:
            print('SUCCESS!')
print('Batch data extraction finished.')


pickle.dump(data['train_inputs'], open('../word2vec/gensim_word2vec_train_inputs.npz', 'wb'))
pickle.dump(data['train_targets'], open('../word2vec/gensim_word2vec_train_targets.npz', 'wb'))
pickle.dump(data['valid_inputs'], open('../word2vec/gensim_word2vec_valid_inputs.npz', 'wb'))
pickle.dump(data['valid_targets'], open('../word2vec/gensim_word2vec_valid_targets.npz', 'wb'))
pickle.dump(data['label_map'], open('../word2vec/gensim_word2vec_label_map.npz', 'wb'))
pickle.dump(data['batched_train_inputs'], open('../word2vec/gensim_word2vec_batched_train_inputs.npz', 'wb'))
pickle.dump(data['batched_train_targets'], open('../word2vec/gensim_word2vec_batched_train_targets.npz', 'wb'))
pickle.dump(data['batched_valid_inputs'], open('../word2vec/gensim_word2vec_batched_valid_inputs.npz', 'wb'))
pickle.dump(data['batched_valid_targets'], open('../word2vec/gensim_word2vec_batched_valid_targets.npz', 'wb'))
