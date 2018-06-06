"""
Tokenize sentence
"""

import nltk


def to_words(sent):
    """
    Tokenize sentence to words
    :param sent: [str], sentences
    :return: [[str word]], word list
    """

    return [nltk.tokenize.word_tokenize(s) for s in sent]


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
        if add_pos:
            w_l = nltk.pos_tag(w_l)
            for word,pos in w_l:
                if by_char:
                    [ret.append(w + pos_prefix + pos) for w in list(word)]
                else:
                    ret.append(word + pos_prefix + pos)
        else:
            if by_char:
                ret = list("".join(w_l))
            else:
                ret = w_l
        ret.append("<END>")

    return ret


def tokenize(sent, by_char=False, add_pos=False, pos_prefix="/"):
    """
    Tokenize sentence
    :param sent: [str], sentences
    :param by_char: bool, if use character as token, default False
    :param add_pos: bool, if do pos tag, default False
    :param pos_prefix: str, character to link word and pos tag, default "/"
    :return: [char token], token list
    """

    word_list = to_words(sent)
    return to_tokens(word_list, by_char, add_pos, pos_prefix)
