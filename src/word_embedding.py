import numpy as np
import re


class word2vector():

    def __init__(self, filename, n_gram):
        self.filename = filename
        self.n_gram = n_gram

    def encode(self):
        lines = open(self.filename, encoding='utf-8').readlines()
        word_set = set()
        for line in lines:
            words = line.split()
            print(words)
            for word in words:
                word_set.add(word)
        word_set.add('UNKNOWN')
        word_set = list(word_set)
        a = open('../data/map', 'w')
        for w in word_set:
            a.write(w+'\n')
        a.close()

        self.word_map = dict()
        for i in range(len(word_set)):
            # print(word_set[i], i)
            self.word_map[word_set[i]] = i
        # print(len(word_set))
        # print(self.word_map)

    def getIndex(self, word):
        if self.word_map.get(word):
            return self.word_map[word]
        else:
            print(word, self.word_map['UNKOWN'])
            return self.word_map['UNKNOWN']


class corpusClean():

    def __init__(self, filename):
        self.filename = filename
        self.num_re = re.compile('[0123456789]+')
        self.char_re = re.compile('[a-zA-Z\-\,\.\。\，]+')

    def clean(self):
        lines = open(self.filename).readlines()
        cleaned = open('../data/cleaned_corpus.txt', 'w', encoding='utf-8')
        for line in lines:
            sentence = line.split()[-1]
            removed_char = re.sub(self.char_re, ' <CHAR> ', sentence)
            removed_num = re.sub(self.num_re, '<NUMBER>', removed_char)
            # if sentence.find('序号')!= -1:
            #     print(line)
            #     print(removed_num)
            #     print(removed_char)
            # break
            cleaned.write(removed_char+'\n')
        cleaned.close()

if __name__ == '__main__':
    # w2v = word2vector('../data/TrainSet.txt', 2)
    # w2v.encode()
    # print(w2v.getIndex('xas'))

    c = corpusClean('../data/TrainSet-eCarX-171019.txt')
    c.clean()
