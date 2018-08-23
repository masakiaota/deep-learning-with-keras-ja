"""
2.5.1 チェックポイント
"""
from __future__ import division, print_function
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
import os

np.random.seed(1671)  # for reproducibility

# network and training
NB_EPOCH = 20
BATCH_SIZE = 128
MODEL_DIR = "./tmp"

# data: shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_train is 60000 rows of 28x28 values --> reshaped in 60000 x 784
RESHAPED = 784
#
X_train = X_train.reshape(60000, RESHAPED).astype("float32")/255
X_test = X_test.reshape(10000, RESHAPED).astype("float32")/255
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


# normalize
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# M_HIDDEN hidden layers
# 10 outputs
# final stage is softmax

model = Sequential()
model.add(Dense(512, input_shape=(RESHAPED,), activation="relu"))
# model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation="relu"))
# model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation="softmax"))
# model.add(Activation('softmax')
model.summary()


model.compile(loss='categorical_crossentropy',
              optimizer="rmsprop",
              metrics=['accuracy'])
# モデルを保存するための準備
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "model-{epoch:02d}.h5")
    #, save_best_only=True #にすることで一番よいモデルだけ保存できる
)

model.fit(X_train, Y_train,
          batch_size=BATCH_SIZE, epochs=NB_EPOCH,
          callbacks=[checkpoint],
          validation_split=1)

score = model.evaluate(X_test, Y_test)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])
