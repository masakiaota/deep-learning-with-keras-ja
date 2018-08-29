"""
3.4.2 学習済みモデルを特徴抽出機として活用する。
"""
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np

base_model = VGG19(weights="imagenet")
# あるモデルの一部だけを持ってくるためには以下の用に記述すればいい
model = Model(inputs=base_model.input,
              outputs=base_model.get_layer("block4_pool").output)
# ここでは、block4_poolと名付けられたレイヤーのoutputだけを抽出している.
model.summary()

img_path = "sample_images_pretrain/elephant.jpg"
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

block4_pool_features = model.predict(x)  # predictとしているが、途中までの中間層表現を取得している
print(block4_pool_features.shape)
