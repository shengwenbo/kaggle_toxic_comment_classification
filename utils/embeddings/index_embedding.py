

class index_embedding(object):
    def __init__(self, vocab_path=".data/vocab.txt", max_vocab_size=50000):
        self._vocab_path = vocab_path
        self._max_vocab_size = max_vocab_size
        self._load_vocab()


    def _load_vocab(self):
        word_list = ["<START>", "<END>", "<OOV>"]
        self.vocab_size = 3

        with open(self._vocab_path, encoding="UTF-8") as fin:
            for line in fin.readlines():
                if line.strip() == "":
                    continue

                word = line.split("\t")[0]
                word_list.append(word)

                self.vocab_size += 1
                if self.vocab_size == self._max_vocab_size:
                    break

        self._w2i = {}
        self._i2w = {}
        for index, word in enumerate(word_list):
            self._i2w[index] = word
            self._w2i[word] = index


    def word_embedding(self, word):
        if word in self._w2i.keys():
            return self._w2i[word]
        else:
            return self._w2i["<OOV>"]

    def sent_embedding(self, sent):
        return [self.word_embedding(w) for w in sent]

