"""
LSTM model
"""
from keras.layers import concatenate,Input,Embedding,SpatialDropout1D,Bidirectional,Conv1D,GlobalAveragePooling1D,GlobalMaxPooling1D,LSTM,Dense
from keras.preprocessing import sequence
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from numpy import random
import os
from tensorflow import logging

from utils.embeddings.index_embedding import index_embedding

class lstm(object):

    def __init__(self, max_len=150, vocab_path="data/vocab.txt", vocab_size=50000, embedding_size=300, optimizer="adam", ckpt_path="log/lstm"):
        self._max_len = max_len
        self._vocab_size = vocab_size
        self._embedding_size = embedding_size
        self._optimizer = optimizer
        self._ckpt_path = ckpt_path
        self._indexing = index_embedding(vocab_path=os.path.abspath(vocab_path), max_vocab_size=vocab_size)

        self._build_model()


    def _build_model(self):
        inp = Input(shape=(self._max_len,))

        x = Embedding(self._vocab_size, self._embedding_size, trainable=True)(inp)
        x = SpatialDropout1D(0.35)(x)

        x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.15, recurrent_dropout=0.15))(x)
        x = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(x)

        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        x = concatenate([avg_pool, max_pool])

        out = Dense(6, activation='sigmoid')(x)

        self._model = Model(inp, out)
        self._model.compile(loss='binary_crossentropy', optimizer=self._optimizer, metrics=['accuracy'])


    def _preprocess(self, sent_list):
        # 转化为下标
        sent_list = [self._indexing.sent_embedding(s) for s in sent_list]

        # 适应输入长度
        sent_list = sequence.pad_sequences(sent_list, maxlen=self._max_len)

        return sent_list


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