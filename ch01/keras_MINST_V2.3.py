"""
1.3.4 隠れ層の追加による精度向上
の演習問題
隠れ層を10つにしてみる

Result

Test score: 0.12716339288232847
Test accuracy: 0.9621

かなりよかった
"""
from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from make_tensorboard import make_tensorboard


np.random.seed(1671)  # for reproducibility

# network and training
NB_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10   # number of outputs = number of digits
OPTIMIZER = SGD()  # optimizer, explained later in this chapter
N_HIDDEN = 128  # 隠れ層のニューロン数
VALIDATION_SPLIT = 0.2  # how much TRAIN is reserved for VALIDATION

# data: shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_train is 60000 rows of 28x28 values --> reshaped in 60000 x 784
RESHAPED = 784
#
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalize
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# M_HIDDEN hidden layers
# 10 outputs
# final stage is softmax
# モデルの構築
model = Sequential()
# どうやら入力層?だけinput_shapeを指定しているらしい
# 層を一つ抜いた
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))
model.add(Activation('relu'))  # 簡単に書けるのが売りのケラスだろうと全結合と発火はやはり交互に重ねないといけないらしい
for _ in range(9):
    model.add(Dense(N_HIDDEN))
    model.add(Activation("relu"))
model.add(Dense(NB_CLASSES))  # 出力を10にする
model.add(Activation('softmax'))
model.summary()

# モデルができたらコンパイル
model.compile(loss='categorical_crossentropy',
              optimizer=OPTIMIZER,
              metrics=['accuracy'])

callbacks = [make_tensorboard(set_dir_name='keras_MINST_V2')]

model.fit(X_train, Y_train,
          batch_size=BATCH_SIZE, epochs=NB_EPOCH,
          callbacks=callbacks,
          verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])
