"""
6.4.1 LSTMで評判分析
Epoch 10/10
5668/5668 [==============================] - 9s 2ms/step - loss: 0.0021 - acc: 0.9995 - val_loss: 0.0459 - val_acc: 0.9880
1418/1418 [==============================] - 0s 281us/step
Test score: 0.046, accuracy: 0.988
1       1       i want to be here because i love harry potter , and i really want a place where people take it serious , but it is still so much fun .
1       1       because i would like to make friends who like the same things i like , and i really like harry potter , so i thought that joining a community like this would be a good start .
1       1       so as felicia 's mom is cleaning the table , felicia grabs my keys and we dash out like freakin mission impossible .
0       0       brokeback mountain was boring .
0       0       not because i hate harry potter , but because i am the type of person that likes it when the main character dies .

"""
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import collections
import os

import nltk
import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import Activation, Dense, Dropout, Embedding, LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import codecs


DATA_DIR = "./data"
LOG_DIR = "./logs"

MAX_FEATURES = 2000
MAX_SENTENCE_LENGTH = 40

EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 10

# Read training data and generate vocabulary
maxlen = 0
word_freqs = collections.Counter()
# """
# import collections

# l = ['a', 'a', 'a', 'a', 'b', 'c', 'c']
# c = collections.Counter(l)

# print(c)
# # Counter({'a': 4, 'c': 2, 'b': 1})

# でも以下の使い方を見る限り、要は辞書を作ってればいいので
# word_freqs={}でも動くのでは？←most_commonのメソッドを使いたかったため
# """
num_recs = 0  # サンプル数に対応
with codecs.open(os.path.join(DATA_DIR, "umich-sentiment-train.txt"), "r",
                 'utf-8') as ftrain:
    for line in ftrain:
        label, sentence = line.strip().split("\t")
        try:
            words = nltk.word_tokenize(sentence.lower())
        except LookupError:
            print("Englisth tokenize does not downloaded. So download it.")
            nltk.download("punkt")
            words = nltk.word_tokenize(sentence.lower())
        maxlen = max(maxlen, len(words))
        for word in words:
            word_freqs[word] += 1
        num_recs += 1

# Get some information about our corpus
print(maxlen)            # 42
print(len(word_freqs))  # 2313
# 自分の環境では>>> print(len(word_freqs)) 2328だった

# 1 is UNK, 0 is PAD
# We take MAX_FEATURES-1 features to account for PAD
# 語彙は2000＋2(UNKとPAD)個しようするものとする
# ?なんでlen(word_freqs)じゃだめなんだろう
vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
# 頻出なのから対応する固有の番号を割り振る
word2index = {x[0]: i+2 for i, x in
              enumerate(word_freqs.most_common(MAX_FEATURES))}
word2index["PAD"] = 0
word2index["UNK"] = 1
index2word = {v: k for k, v in word2index.items()}

# 今は文章の羅列であるこれを数字の羅列に変換する
# convert sentences to sequences
X = np.empty((num_recs, ), dtype=list)
y = np.zeros((num_recs, ))
i = 0
with codecs.open(os.path.join(DATA_DIR, "umich-sentiment-train.txt"),
                 'r', 'utf-8') as ftrain:
    for line in ftrain:
        label, sentence = line.strip().split("\t")
        words = nltk.word_tokenize(sentence.lower())
        seqs = []
        for word in words:
            # もし辞書に単語があったら、対応する数字に直して、なかったらUNKにする.
            if word in word2index:
                seqs.append(word2index[word])
            else:
                seqs.append(word2index["UNK"])
        X[i] = seqs  # i番目のサンプルの文章に対応する数字の配列を作成
        y[i] = int(label)
        i += 1

# Pad the sequences (left padded with zeros)
# 左側に0を埋める
X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)

# Split input into training and test
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2,
                                                random_state=42)
print(Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)

# Build model
model = Sequential()
# ここでの入力テンソルのshapeは(None, MAX_SENTENCE_LENGTH, 1)
# 第一次元はバッチサイズに対応、指定なしなのでNone
# 第二次元は時系列方向の長さに対応
# 第三次元はある時刻の入力の要素数に対応(今はある数字が入っているだけなので1)

# 埋め込みword2vec?
model.add(Embedding(vocab_size, EMBEDDING_SIZE,
                    input_length=MAX_SENTENCE_LENGTH))
# 分散表現を獲得したことで、第三次元が変化
# 入力テンソルは (None, MAX_SENTENCE_LENGTH, EMBEDDING_SIZE)となる。
model.add(Dropout(0.5))
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.5, recurrent_dropout=0.5))
# LSTMの出力サイズはreturn_sequence=Trueで(None, HIDDEN_LAYER_SIZE, MAX_SENTENCE_LENGTH)になる
# Falseの場合は(None, HIDDEN_LAYER_SIZE)となる。デフォではこっち（今回もこっち）
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam",
              metrics=["accuracy"])


if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

history = model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE,
                    epochs=NUM_EPOCHS,
                    callbacks=[TensorBoard(LOG_DIR)],
                    validation_data=(Xtest, ytest))

# evaluate
score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
print("Test score: {:.3f}, accuracy: {:.3f}".format(score, acc))

for i in range(5):
    idx = np.random.randint(len(Xtest))
    xtest = Xtest[idx].reshape(1, 40)
    ylabel = ytest[idx]
    ypred = model.predict(xtest)[0][0]
    sent = " ".join([index2word[x] for x in xtest[0].tolist() if x != 0])
    print("{:.0f}\t{:.0f}\t{}".format(ypred, ylabel, sent))
