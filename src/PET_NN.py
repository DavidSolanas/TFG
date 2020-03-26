import os
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator

local_weights_file = '/Users/david/Documents/TFG/weights/inception_v3_weights.h5'

pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights=None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
    layer.trainable = False

pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed10')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

# Flatten the output layer to 1 dimension
x = layers.Dropout(rate=0.4)(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(3, activation='relu')(x)
# Add a final sigmoid layer for classification
x = layers.Dense(3, activation='softmax')(x)

model = Model(pre_trained_model.input, x)

model.compile(optimizer=Adam(lr=0.0001),
              loss='categorical_crossentropy',
              metrics=['acc'])

# Define our example directories and files
base_dir = '/Users/david/TFG/brain_data'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_AD_dir = os.path.join(train_dir, 'AD')  # Directory with our training cat pictures
train_CN_dir = os.path.join(train_dir, 'CN')  # Directory with our training cat pictures
train_MCI_dir = os.path.join(train_dir, 'MCI')  # Directory with our training dog pictures

validation_AD_dir = os.path.join(validation_dir, 'AD')  # Directory with our training cat pictures
validation_CN_dir = os.path.join(validation_dir, 'CN')  # Directory with our training cat pictures
validation_MCI_dir = os.path.join(validation_dir, 'MCI')  # Directory with our training dog pictures

train_AD_fnames = os.listdir(train_AD_dir)
train_CN_fnames = os.listdir(train_CN_dir)
train_MCI_fnames = os.listdir(train_MCI_dir)

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.08)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator()

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=8,
                                                    class_mode='categorical',
                                                    target_size=(512, 512))

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        batch_size=8,
                                                        class_mode='categorical',
                                                        target_size=(512, 512))

history = model.fit_generator(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=100,
    epochs=20,
    validation_steps=50,
    verbose=1)
