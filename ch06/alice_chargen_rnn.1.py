# -*- coding: utf-8 -*-
"""
6.1.1RNNを用いたテキスト生成
をLSTMに置き換えて行ってみる
 
結果は以下のようになったが
Iteration #: 23
Epoch 1/1
162739/162739 [==============================] - 24s 148us/step - loss: 1.0903
Generating from seed: yourself,’
yourself,’ said the dormouse to see if you distaic warch it out of the works to say in a low to some of the so
==================================================
Iteration #: 24
Epoch 1/1
162739/162739 [==============================] - 25s 155us/step - loss: 1.0784
Generating from seed: m a poor m
m a poor manted the gryphon and don’t be a lew the caterpillar said to herself, ‘i don’t know what the mouse w

"""
from __future__ import print_function

import numpy as np
from keras.layers import Dense, Activation, SimpleRNN, LSTM
from keras.models import Sequential
import codecs


INPUT_FILE = "./data/alice_in_wonderland.txt"

# extract the input as a stream of characters
# 改行と非アスキー文字をクリーンアップして、内容をtextという変数に保存
print("Extracting text from input...")
with codecs.open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = [line.strip().lower() for line in f
             if len(line) != 0]
    text = " ".join(lines)
    # >>> "@@@@@@@@".join(lines[:4])
    # u'\ufeffproject gutenberg\u2019s alice\u2019s adventures in wonderland, by lewis carroll@@@@@@@@@@@@@@@@this ebook is for the use of anyone anywhere at no cost and with@@@@@@@@almost no restrictions whatsoever.  you may copy it, give it away or'


# creating lookup tables ルックアップテーブルの作成
# これより文字ではなく文字のインデックスについて扱うのでその準備
# Here chars is the number of features in our character "vocabulary"
chars = set(text)
nb_chars = len(chars)  # 60だった
char2index = dict((c, i) for i, c in enumerate(chars))
index2char = dict((i, c) for i, c in enumerate(chars))
# >>> dict((c, i) for i, c in enumerate(["a", "b", "c"]))
# {'a': 0, 'c': 2, 'b': 1}


# create inputs and labels from the text. We do this by stepping
# through the text ${step} character at a time, and extracting a
# sequence of size ${seqlen} and the next output char. For example,
# assuming an input text "The sky was falling", we would get the
# following sequence of input_chars and label_chars (first 5 only)
#   The sky wa -> s
#   he sky was ->
#   e sky was  -> f
#    sky was f -> a
#   sky was fa -> l
print("Creating input and label text...")
SEQLEN = 10
STEP = 1

input_chars = []
label_chars = []
for i in range(0, len(text) - SEQLEN, STEP):
    input_chars.append(text[i:i + SEQLEN])
    label_chars.append(text[i + SEQLEN])

# vectorize the input and label chars
# Each row of the input is represented by seqlen characters, each
# represented as a 1-hot encoding of size len(char). There are
# len(input_chars) such rows, so shape(X) is (len(input_chars),
# seqlen, nb_chars).
# Each row of output is a single character, also represented as a
# dense encoding of size len(char). Hence shape(y) is (len(input_chars),
# nb_chars).
# 文字をonehotにすることを考える。
# まず入力一つあたりはSEQLEN（10）分の文字列である。
# 語彙数はnb_charsである(onehotの要素数になる)
# したがって各入力のテンソルサイズは(SEQLEN, nb_chars)となる
# さらにすべての入力がいくつかと考えるとlen(input_chars)である。
print("Vectorizing input and label text...")
X = np.zeros((len(input_chars), SEQLEN, nb_chars), dtype=np.bool)
y = np.zeros((len(input_chars), nb_chars), dtype=np.bool)
for i, input_char in enumerate(input_chars):
    for j, ch in enumerate(input_char):
        # ある一文字に対応するindexのところだけ1にする
        X[i, j, char2index[ch]] = 1
    y[i, char2index[label_chars[i]]] = 1

# モデルの構築
# Build the model. We use a single RNN with a fully connected layer
# to compute the most likely predicted output char
HIDDEN_SIZE = 128
BATCH_SIZE = 128
NUM_ITERATIONS = 25
NUM_EPOCHS_PER_ITERATION = 1
NUM_PREDS_PER_EPOCH = 100

model = Sequential()
# 系列から1つだけを返してもらいたいので、return_sequences=Falseである。
# tensorflowのパフォーマンス向上のためunroll=Trueらしい
model.add(LSTM(HIDDEN_SIZE, return_sequences=False,  # ここをSimpleRNNからLSTMに置換しただけ
               input_shape=(SEQLEN, nb_chars),
               unroll=True))
# RNNは隠れ層扱いで、最終的に全結合層に受け渡さないとニューロンの数があれ（語彙力）
model.add(Dense(nb_chars))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
model.summary()

# 今までと訓練のアプローチを変える
# 1EPOCHずつ学習させてテストする。これをNUM_ITERATIONS(25回)繰り返し、人間に理解出来る出力で停止する
# テストは本当はEPOCHやるたびにしないといけないのかな？だけどここでは、横着してNUM_ITERATIONIS回してからやることにする
# We train the model in batches and test output generated at each step
for iteration in range(NUM_ITERATIONS):
    print("=" * 50)
    print("Iteration #: {}".format(iteration))
    # 続きからfitできるのか
    model.fit(X, y, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS_PER_ITERATION)

    # testing model
    # randomly choose a row from input_chars, then use it to
    # generate text from model for next 100 chars
    test_idx = np.random.randint(len(input_chars))
    # 適当に選んだ10文字を与えて次の1文字を予測する
    # その1文字を加えて更に次の1文字を予測（長さを保つために最初に1文字は消したが）
    test_chars = input_chars[test_idx]
    print("Generating from seed: {}".format(test_chars))
    print(test_chars, end="")
    for i in range(NUM_PREDS_PER_EPOCH):
        Xtest = np.zeros((1, SEQLEN, nb_chars))
        for i, ch in enumerate(test_chars):
            Xtest[0, i, char2index[ch]] = 1
        pred = model.predict(Xtest, verbose=0)[0]
        ypred = index2char[np.argmax(pred)]
        print(ypred, end="")
        # move forward with test_chars + ypred
        test_chars = test_chars[1:] + ypred
    print()
