"""6.5.1 GRUで品詞タグづけ

GRU
Epoch 1/1
3131/3131 [==============================] - 56s 18ms/step - loss: 1.0662 - acc: 0.9060 - val_loss: 0.5404 - val_acc: 0.9159
783/783 [==============================] - 3s 3ms/step
Test score: 0.540, accuracy: 0.916
Train on 3131 samples, validate on 783 samples

LSTM
Epoch 1/1
3131/3131 [==============================] - 65s 21ms/step - loss: 0.9938 - acc: 0.9036 - val_loss: 0.5494 - val_acc: 0.9159
783/783 [==============================] - 3s 3ms/step
Test score: 0.549, accuracy: 0.916
Train on 3131 samples, validate on 783 samples

bidirectional LSTMjj
Epoch 1/1
3131/3131 [==============================] - 90s 29ms/step - loss: 0.8276 - acc: 0.9075 - val_loss: 0.4836 - val_acc: 0.9159
783/783 [==============================] - 4s 5ms/step
Test score: 0.484, accuracy: 0.916


"""
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import collections
import os

import nltk
import numpy as np
from keras.layers import Activation, Dense, Dropout, RepeatVector, Embedding, \
    GRU, LSTM, TimeDistributed, Bidirectional
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


def parse_sentences(filename):
    word_freqs = collections.Counter()
    num_recs, maxlen = 0, 0
    with open(filename, "r") as fin:
        for line in fin:
            words = line.strip().lower().split()
            for word in words:
                word_freqs[word] += 1
            maxlen = max(maxlen, len(words))
            num_recs += 1
    return word_freqs, maxlen, num_recs


def build_tensor(filename, numrecs, word2index, maxlen):
    data = np.empty((numrecs, ), dtype=list)
    with open(filename, "r") as fin:
        for i, line in enumerate(fin):
            wids = []
            for word in line.strip().lower().split():
                if word in word2index:
                    wids.append(word2index[word])
                else:
                    wids.append(word2index["UNK"])
            data[i] = wids
    pdata = sequence.pad_sequences(data, maxlen=maxlen)
    return pdata


DATA_DIR = "./data"

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
with open(os.path.join(DATA_DIR, "treebank_sents.txt"), "w") as fedata, \
        open(os.path.join(DATA_DIR, "treebank_poss.txt"), "w") as ffdata:
    sents = nltk.corpus.treebank.tagged_sents()
    for sent in sents:
        words, poss = [], []
        for word, pos in sent:
            if pos == "-NONE-":
                continue
            words.append(word)
            poss.append(pos)
        fedata.write("{:s}\n".format(" ".join(words)))
        ffdata.write("{:s}\n".format(" ".join(poss)))


s_wordfreqs, s_maxlen, s_numrecs = \
    parse_sentences(os.path.join(DATA_DIR, "treebank_sents.txt"))
t_wordfreqs, t_maxlen, t_numrecs = \
    parse_sentences(os.path.join(DATA_DIR, "treebank_poss.txt"))
print("# records: {:d}".format(s_numrecs))
print("# unique words: {:d}".format(len(s_wordfreqs)))
print("# unique POS tags: {:d}".format(len(t_wordfreqs)))
print("# words/sentence: max: {:d}".format(s_maxlen))


MAX_SEQLEN = 250
S_MAX_FEATURES = 5000
T_MAX_FEATURES = 45


s_vocabsize = min(len(s_wordfreqs), S_MAX_FEATURES) + 2
s_word2index = {x[0]: i+2 for i, x in
                enumerate(s_wordfreqs.most_common(S_MAX_FEATURES))}
s_word2index["PAD"] = 0
s_word2index["UNK"] = 1
s_index2word = {v: k for k, v in s_word2index.items()}

t_vocabsize = len(t_wordfreqs) + 1
t_word2index = {x[0]: i for i, x in
                enumerate(t_wordfreqs.most_common(T_MAX_FEATURES))}
t_word2index["PAD"] = 0
t_index2word = {v: k for k, v in t_word2index.items()}


X = build_tensor(os.path.join(DATA_DIR, "treebank_sents.txt"),
                 s_numrecs, s_word2index, MAX_SEQLEN)
Y = build_tensor(os.path.join(DATA_DIR, "treebank_poss.txt"),
                 t_numrecs, t_word2index, MAX_SEQLEN)
Y = np.array([np_utils.to_categorical(d, t_vocabsize) for d in Y])
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y,
                                                test_size=0.2, random_state=42)


EMBED_SIZE = 128
HIDDEN_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 1

# GRU
print()
print("GRU")
# (None, AX_SEQLEN, 1)
model = Sequential()
model.add(Embedding(s_vocabsize, EMBED_SIZE, input_length=MAX_SEQLEN))
# (None, MAX_SEQLEN, EMBED_SIZE)
model.add(Dropout(0.2))

# EncoderRNNであり、
# return_sequeces=False
model.add(GRU(HIDDEN_SIZE, dropout=0.2, recurrent_dropout=0.2))
# (None, HIDDEN_SIZE)
# RepeatVector層は前の入力を複製する役割の層らしい
model.add(RepeatVector(MAX_SEQLEN))
# (None, MAX_SEQLEN, HIDDEN_SIZE)
# デコーダーRNN 今までの状態とEncoderRNNの内部状態に基づいて品詞ごとに内部状態を変える
model.add(GRU(HIDDEN_SIZE, return_sequences=True))
# (None, MAX_SEQLEN, HIDDEN_SIZE)
# TimeDistributedがわからん　詳しくはhttps://keras.io/ja/layers/wrappers/
# このラッパーにより，入力のすべての時間スライスにレイヤーを適用できます．
# 時間的な順序を残すときに使うっぽい？
model.add(TimeDistributed(Dense(t_vocabsize)))
# (None, MAX_SEQLEN, t_vocabsize) 時系列に対応するMAX_SEQLENが残っている。
model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE,
          epochs=NUM_EPOCHS, validation_data=[Xtest, Ytest])
score, acc = model.evaluate(Xtest, Ytest, batch_size=BATCH_SIZE)
print("Test score: {:.3f}, accuracy: {:.3f}".format(score, acc))


# LSTM
print()
print("LSTM")
model = Sequential()
model.add(Embedding(s_vocabsize, EMBED_SIZE, input_length=MAX_SEQLEN))
model.add(Dropout(0.2))
model.add(LSTM(HIDDEN_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(RepeatVector(MAX_SEQLEN))
model.add(LSTM(HIDDEN_SIZE, return_sequences=True))
model.add(TimeDistributed(Dense(t_vocabsize)))
model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])
model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE,
          epochs=NUM_EPOCHS, validation_data=[Xtest, Ytest])
score, acc = model.evaluate(Xtest, Ytest, batch_size=BATCH_SIZE)
print("Test score: {:.3f}, accuracy: {:.3f}".format(score, acc))


# Bidirectional LSTM
print()
print("Bidictional LSTM")
model = Sequential()
model.add(Embedding(s_vocabsize, EMBED_SIZE, input_length=MAX_SEQLEN))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(HIDDEN_SIZE, dropout=0.2, recurrent_dropout=0.2)))
model.add(RepeatVector(MAX_SEQLEN))
model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True)))
model.add(TimeDistributed(Dense(t_vocabsize)))
model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])
model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE,
          epochs=NUM_EPOCHS, validation_data=[Xtest, Ytest])
score, acc = model.evaluate(Xtest, Ytest, batch_size=BATCH_SIZE)
print("Test score: {:.3f}, accuracy: {:.3f}".format(score, acc))
