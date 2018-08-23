"""
P9 1.3実例手書き数字認識に対応する。
流れとしては
- 定数の宣言
- データの準備
- modelの構造
- モデルのコンパイル
- モデルの訓練
- 評価
である。 
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
# 定数の宣言
NB_EPOCH = 200
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10   # number of outputs = number of digits
OPTIMIZER = SGD()  # SGD optimizer, explained later in this chapter
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2  # how much TRAIN is reserved for VALIDATION

# data: shuffled and split between train and test sets
# データの準備
# train 60000サンプル, test10000サンプル
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_train is 60000 rows of 28x28 values --> reshaped in 60000 x 784
RESHAPED = 784
#
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalize
#
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# いまy_trainはarray([5, 0, 4, ..., 5, 6, 8], dtype=uint8)みたいになっている
# それを
"""
array([[0., 0., 0., ..., 0., 0., 0.],
       [1., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 1., 0.]], dtype=float32)みたいなone-hotにする関数
"""
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# 10 outputs
# final stage is softmax
# モデルの構築 全結合の入力層と出力層があるのみ
model = Sequential()
# Dense(出力のニューロン数, inputの数)
model.add(Dense(NB_CLASSES, input_shape=(RESHAPED,)))
model.add(Activation('softmax'))

print("model summary")
model.summary()

# 損失関数の設定、最適化の設定、評価基準はコンパイルのときに指定する
model.compile(loss='categorical_crossentropy',
              optimizer=OPTIMIZER,
              metrics=['accuracy'])

callbacks = [make_tensorboard(set_dir_name='keras_MINST_V1')]

model.fit(X_train, Y_train,
          batch_size=BATCH_SIZE, epochs=NB_EPOCH,
          callbacks=callbacks,  # callbackは読んでも良うわからんかった
          verbose=VERBOSE, validation_split=VALIDATION_SPLIT)  # 検証用にtrainの中の0.2を用いる

score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])
