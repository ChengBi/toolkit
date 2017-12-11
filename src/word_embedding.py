class wordEmbedding():

    def __init__(self, filename, n_gram):
        self.filename = filename
        self.n_gram = n_gram

    def loadCorpus(self):
        lines = open(self.filename).readlines()
        word_set = set()
        for line in lines:
            words = line.split()
            for word in words:
                word_set.add(word)
        word_set.add('UNKNOWN')

        
