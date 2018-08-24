"""
3.2.1kerasによるLenetの実装

48000/48000 [==============================] - 66s 1ms/step - loss: 0.0064 - acc: 0.9979 - val_loss: 0.0421 - val_acc: 0.9894
Test loss: 0.03667050626840614
Test accuracy: 0.9915

"""
import os
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.callbacks import TensorBoard


def lenet(input_shape, num_classes):
    """
    Lenetを定義する関数
    """
    model = Sequential()

    # extract image features by convolution and max pooling layers
    # フィルターを20枚用意, 小窓のサイズ5×5, paddingによって入力と出力の画像サイズは同じ
    model.add(Conv2D(
        20, kernel_size=5, padding="same",
        input_shape=input_shape, activation="relu"
    ))
    # 2, 2でマックスプーリング
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 再度畳み込み、深い層ほどフィルターを増やすのはテクニック
    model.add(Conv2D(50, kernel_size=5, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # classify the class by fully-connected layers
    # Flatten()はマトリックスを1次元ベクトルに変換する層
    # FCにつなぐために必要
    model.add(Flatten())
    model.add(Dense(500, activation="relu"))
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))
    return model


class MNISTDataset():
    """
    便利なメソッドを追加したクラスにしてしまう
    """

    def __init__(self):
        self.image_shape = (28, 28, 1)  # image is 28x28x1 (grayscale)
        self.num_classes = 10

    def get_batch(self):
        """
        バッチというかうまく処理して返してるだけな気がするけど、

        内包表記で見慣れない書き方があるが
        > a,b=[(x) for x in [(1,2),(3,4)]]
        >>> a
        (1, 2)
        >>> b
        (3, 4)
        って感じ
        """
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = [self.preprocess(d) for d in [x_train, x_test]]
        y_train, y_test = [self.preprocess(d, label_data=True) for d in
                           [y_train, y_test]]

        return x_train, y_train, x_test, y_test

    def preprocess(self, data, label_data=False):
        """
        labelをone-hotにして、データを0-1に正規化する。
        """
        if label_data:
            # convert class vectors to binary class matrices
            data = keras.utils.to_categorical(data, self.num_classes)
        else:
            data = data.astype("float32")
            data /= 255  # convert the value to 0~1 scale
            shape = (data.shape[0],) + self.image_shape  # add dataset length
            data = data.reshape(shape)

        return data


class Trainer():
    """
    ネットワークに学習させるものまでクラスにしてしまう。
    """

    def __init__(self, model, loss, optimizer):
        """
        model ... kerasのモデル
        loss ... 損失関数
        optimizer ... 最適化手法
        を受け取り準備する。
        """
        self._target = model
        # initでcompileするのが確かに楽そうだ
        self._target.compile(
            loss=loss, optimizer=optimizer, metrics=["accuracy"]
        )
        self.verbose = 1
        logdir = "logdir_lenet"
        self.log_dir = os.path.join(os.path.dirname(__file__), logdir)

    def train(self, x_train, y_train, batch_size, epochs, validation_split):
        """
        学習の定義
        """

        # logを保存するdirの操作、もしすでにあったら削除して新たにつくる。
        if os.path.exists(self.log_dir):
            import shutil
            shutil.rmtree(self.log_dir)  # remove previous execution
            """
            shutil.rmtree(path, ignore_errors=False, onerror=None)(原文)
            ディレクトリツリー全体を削除します。 path はディレクトリを指していなければなりません (ただしディレクトリに対するシンボリックリンクではいけません)。
            """
        os.mkdir(self.log_dir)

        # 学習
        self._target.fit(
            x_train, y_train,
            batch_size=batch_size, epochs=epochs,
            validation_split=validation_split,
            # 先程作ったlod_dirにtensor boardの結果が無事保存される。
            callbacks=[TensorBoard(log_dir=self.log_dir)],
            verbose=self.verbose
        )


dataset = MNISTDataset()

# make model
model = lenet(dataset.image_shape, dataset.num_classes)

# train the model
x_train, y_train, x_test, y_test = dataset.get_batch()
trainer = Trainer(model, loss="categorical_crossentropy", optimizer=Adam())
trainer.train(
    x_train, y_train, batch_size=128, epochs=12, validation_split=0.2
)

# show result
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
