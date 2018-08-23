"""
1.3.5 ドロップアウトによる精度向上

EPOCHが20のときは
loss: 0.2931 - acc: 0.9131 - val_loss: 0.1973 - val_acc: 0.9421
Test score: 0.19933920312449335
Test accuracy: 0.9403
だった、学習データのほうが精度が出てないのは学習しきっていない可能性がある。

EPOCHを250にかえてみる
loss: 0.0616 - acc: 0.9808 - val_loss: 0.0802 - val_acc: 0.9777
Test score: 0.07741344122756273
Test accuracy: 0.9781

Epochを増やしたら十分学習できた！

"""
from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation  # dropoutをimportする。
from keras.optimizers import SGD
from keras.utils import np_utils
from make_tensorboard import make_tensorboard


np.random.seed(1671)  # for reproducibility

# network and training
NB_EPOCH = 250
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10   # number of outputs = number of digits
OPTIMIZER = SGD()  # optimizer, explained later in this chapter
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2  # how much TRAIN is reserved for VALIDATION
DROPOUT = 0.3  # ここでドロップするニューロンの割合を指定している。

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

model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))  # dropoutを活性化関数のあとに挿入するのがお作法
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=OPTIMIZER,
              metrics=['accuracy'])

callbacks = [make_tensorboard(set_dir_name='keras_MINST_V3')]

model.fit(X_train, Y_train,
          batch_size=BATCH_SIZE, epochs=NB_EPOCH,
          callbacks=callbacks,
          verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])
