"""
From tokenized sentences to vocabulary
"""

import os
from collections import Counter


def load_vocab(path, encoding="UTF-9"):
    """
    Load vocabulary from file
    :param path: str, file path
    :param encoding: str, file encoding, default "UTF-8"
    :return: [(str word: int, freq)], vaocabulary info
    """
    vocab = []

    if not os.path.exists(path):
        return vocab

    with open(path, encoding=encoding) as fin:
        for line in fin.readlines():
            line = line.strip()
            word, freq = line.split("\t")
            vocab.append((word,int(freq)))

    return vocab


def dump_vocab(vocab, path, encoding="Utf-8"):
    """
    Dump vocabulary to file
    :param vocab: {str word: int, freq}, vaocabulary info
    :param path:  str, file path
    :param encoding: str, file encoding, default "UTF-8"
    :return: None
    """
    with open(path, "w", encoding=encoding) as fout:
        for word, freq in vocab:
            fout.write("%s\t%d\n" % (word, freq))


def make_vocab(sent_list):
    """
    Stat words in sentences, make vocabulary
    :param sent_list: [[str token]], sentence list
    :return: [(str word: int, freq)], vaocabulary info
    """
    counter = Counter()
    for sent in sent_list:
        counter.update(sent)

    ret = list(counter.items())
    ret.sort(key=lambda x: x[1], reverse=True)
    return ret
