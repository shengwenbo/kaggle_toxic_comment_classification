"""
Tokenize sentence
"""

import nltk
from nltk.corpus import stopwords
from utils.split_sentence import split_into_sentences

stop_words = stopwords.words("english")

def to_words(sents):
    """
    Tokenize sentence to words
    :param sents: [str], sentences
    :return: [[str word]], word list
    """

    ret = []
    for sent in sents:
        sent_seg = []
        for w in nltk.tokenize.word_tokenize(sent):
            if w not in stop_words and len(w) > 1:
                sent_seg.append(w)
        ret.append(sent_seg)

    return ret


def to_tokens(word_list, by_char=False, add_pos=False, pos_prefix="/"):
    """
    Word list to tokens
    :param word_list: [[str word]], word list
    :param by_char: bool, if use character as token, default False
    :param add_pos: bool, if do pos tag, default False
    :param pos_prefix: str, character to link word and pos tag, default "/"
    :return: [char token], token list
    """

    ret = []
    for w_l in word_list:
        sent_seg = []
        if add_pos:
            w_l = nltk.pos_tag(w_l)
            for word,pos in w_l:
                if by_char:
                    [sent_seg.append(w + pos_prefix + pos) for w in list(word)]
                else:
                    sent_seg.append(word + pos_prefix + pos)
        else:
            if by_char:
                sent_seg = list("".join(w_l))
            else:
                sent_seg = w_l
        sent_seg.append("<END>")
        ret += sent_seg

    return ret


def tokenize(sent, by_char=False, add_pos=False, pos_prefix="/"):
    """
    Tokenize sentence
    :param sent: str, sentences
    :param by_char: bool, if use character as token, default False
    :param add_pos: bool, if do pos tag, default False
    :param pos_prefix: str, character to link word and pos tag, default "/"
    :return: [char token], token list
    """
    sents = split_into_sentences(sent)
    word_list = to_words(sents)
    return to_tokens(word_list, by_char, add_pos, pos_prefix)
