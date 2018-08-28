"""
3.3.3 学習したモデルを利用し予測する

cifar10_deep_with_aug.pyで作ったモデルをlogdirに保存してあるのでそれを読み出して、推論を行う
"""
from pathlib import Path
import numpy as np
from PIL import Image
from keras.models import load_model


model_path = "logdir_cifar10_deep_with_aug/model_file.hdf5"
images_folder = "sample_images"

# load model
model = load_model(model_path)
image_shape = (32, 32, 3)


# load images
def crop_resize(image_path):
    """
    入力のサイズ(32,32,3)に合わせるための加工
    """
    image = Image.open(image_path)
    length = min(image.size)  # 短い方の辺に合わせてるのかな？
    crop = image.crop((0, 0, length, length))  # 0pixelからlength pixelまでトリミング
    resized = crop.resize(image_shape[:2])  # 32,32に
    img = np.array(resized).astype("float32")
    img /= 255
    return img


folder = Path(images_folder)
# image_paths = [str(f) for f in folder.glob("*.png")]
image_paths = list(folder.glob("*.png"))
images = [crop_resize(p) for p in image_paths]  # この時点ではまだ一番外の次元がリスト
images = np.array(images)  # (2, 32, 32, 3)のndarrayになった

predicted = model.predict_classes(images)

assert predicted[0] == 3, "image should be cat."  # 条件式がTrueではない時に、例外を投げます。
# 詳しくはhttps://qiita.com/nannoki/items/15004992b6bb5637a9cd
assert predicted[1] == 5, "image should be dog."

print("You can detect cat & dog!")
