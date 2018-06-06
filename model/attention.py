"""
LSTM + Attention model
"""
from keras.layers import concatenate,Input,Embedding,SpatialDropout1D,Bidirectional,Conv1D,GlobalAveragePooling1D,GlobalMaxPooling1D,LSTM,Dense,Permute,Flatten
from keras.layers.core import RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing import sequence
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from numpy import random
import os
from tensorflow import logging
from model.attention_layer import AttLayer
import numpy as np

from utils.embeddings.index_embedding import index_embedding

class attention(object):

    def __init__(self, max_sents=15, max_len=150, vocab_path="data/vocab.txt", vocab_size=50000, embedding_size=300, optimizer="adam", ckpt_path="log/attention"):
        self._max_sents = max_sents
        self._max_len = max_len
        self._vocab_size = vocab_size
        self._embedding_size = embedding_size
        self._optimizer = optimizer
        self._ckpt_path = ckpt_path
        self._indexing = index_embedding(vocab_path=os.path.abspath(vocab_path), max_vocab_size=vocab_size)

        self._build_model()


    def _build_model(self):
        embedding_layer = Embedding(self._vocab_size, self._embedding_size, trainable=True)

        sentence_input = Input(shape=(self._max_len,))
        embedded_sequences = embedding_layer(sentence_input)
        l_lstm = Bidirectional(LSTM(128))(embedded_sequences)
        sentEncoder = Model(sentence_input, l_lstm)

        review_input = Input(shape=(15, self._max_len), dtype='int32')
        review_encoder = TimeDistributed(sentEncoder)(review_input)
        l_lstm_sent = Bidirectional(LSTM(128))(review_encoder)
        preds = Dense(2, activation='sigmoid')(l_lstm_sent)
        self._model = Model(review_input, preds)

        self._model.compile(loss='categorical_crossentropy', optimizer=self._optimizer, metrics=['accuracy'])


    def _preprocess(self, doc_list):
        doc_ind_list = []
        for doc in doc_list:
            # 分句
            doc_s = " ".join(doc)
            sents_s = doc_s.split("<END>")
            sents = [s.strip().split(" ") for s in sents_s]
            # 转化为下标
            sents = [self._indexing.sent_embedding(s) for s in sents]
            # 适应输入长度
            sents = sequence.pad_sequences(sents, maxlen=self._max_len)
            doc_ind_list.append(sents)

        doc_list = np.vstack(tuple(doc_ind_list))
        doc_list = sequence.pad_sequences(doc_list, maxlen=self._max_sents)
        return doc_list


    def restore(self):
        ckpt_path = os.path.join(self._ckpt_path, "best_weights.hdf5")
        if os.path.exists(ckpt_path):
            logging.info("restoring model from %s" % ckpt_path)
            self._model.load_weights(os.path.join(self._ckpt_path, "best_weights.hdf5"))
            return True
        else:
            return False


    def train(self, train_x, train_y, batch_size=32, validation_split=0.1, epochs=10):
        train_x = self._preprocess(train_x)

        # 打乱数据
        index = [i for i in range(len(train_x))]
        random.shuffle(index)
        train_x = train_x[index]
        train_y = train_y[index]

        self._model.fit(train_x, train_y,
                        batch_size=batch_size, epochs=epochs,
                        validation_split=validation_split, shuffle=True,
                        callbacks=[ReduceLROnPlateau(patience=1),
                                   ModelCheckpoint(filepath=os.path.join(self._ckpt_path,"best_weights.hdf5"), save_best_only=True),
                                   TensorBoard(log_dir=self._ckpt_path)],
                        verbose=1)


    def evaluate(self, test_x, test_y, batch_size=32):
        test_x = self._preprocess(test_x)

        return zip(self._model.metrics_names, self._model.evaluate(test_x, test_y, batch_size=batch_size, verbose=1))


    def predict(self, pred_x, batch_size=32):
        pred_x = self._preprocess(pred_x)

        return self._model.predict(pred_x, batch_size=batch_size, verbose=1)