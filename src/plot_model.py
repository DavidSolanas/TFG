from keras.utils import plot_model
import keras.applications as apps

model = apps.inception_v3.InceptionV3(include_top=False)
# plot_model(model, to_file='model.png')

for i, layer in enumerate(model.layers[:18]):
    print(i, layer.name)
