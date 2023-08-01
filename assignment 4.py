# Do not change this code
try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass

# Do not change this code
# pip install --upgrade --no-cache-dir gdown

# Do not change this code
# !gdown "https://storage.googleapis.com/learning-datasets/beans/train.zip" -O /tmp/train.zip
# !gdown "https://storage.googleapis.com/learning-datasets/beans/validation.zip" -O /tmp/validation.zip
# !gdown "https://storage.googleapis.com/learning-datasets/beans/test.zip" -O /tmp/test.zip

# Do not change this code
import os
import zipfile

local_zip = '/tmp/train.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
local_zip = '/tmp/validation.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
local_zip = '/tmp/test.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/test')
zip_ref.close()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Complete the code given below
# train_datagen = ImageDataGenerator(
#      # YOUR CODE HERE #
# )

# validation_datagen = ImageDataGenerator(
#      # YOUR CODE HERE #
# )

# TRAIN_DIRECTORY_LOCATION = # YOUR CODE HERE #
# VAL_DIRECTORY_LOCATION = # YOUR CODE HERE #
# TARGET_SIZE = # YOUR CODE HERE #
# CLASS_MODE = # YOUR CODE HERE #

# Solution
train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest'

)

TRAIN_DIRECTORY_LOCATION = '/tmp/train'

VAL_DIRECTORY_LOCATION = '/tmp/validation'

TARGET_SIZE = (224,224)

CLASS_MODE = 'categorical'

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIRECTORY_LOCATION,
    target_size = TARGET_SIZE,  
    batch_size = 128,
    class_mode = CLASS_MODE
)

validation_generator = validation_datagen.flow_from_directory(
    VAL_DIRECTORY_LOCATION,
    target_size = TARGET_SIZE,  
    batch_size = 128,
    class_mode = CLASS_MODE
)

import tensorflow as tf
model = tf.keras.models.Sequential([
    # Find the features with Convolutions and Pooling

   tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(224, 224, 3)),

   tf.keras.layers.MaxPooling2D(2, 2),

   tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

   tf.keras.layers.MaxPooling2D(2,2),

   tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

   tf.keras.layers.MaxPooling2D(2,2),

   tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

   tf.keras.layers.MaxPooling2D(2,2),

   # Flatten the results to feed into a DNN

   tf.keras.layers.Flatten(),

   # 512 neuron hidden layer

   tf.keras.layers.Dense(512, activation='relu'),

   tf.keras.layers.Dense(3, activation='softmax')
])

# This will print a summary of your model when you're done!
model.summary()

# The following optimizer works well as well
# from tensorflow.keras.optimizers import RMSprop
# OPTIMIZER = RMSprop(lr=0.0001)

OPTIMIZER = 'adam'
LOSS_FUNCTION = 'categorical_crossentropy'

model.compile(
    loss = LOSS_FUNCTION,
    optimizer = OPTIMIZER,
    metrics = ['accuracy']
)

NUM_EPOCHS = 20 

history = model.fit(
      train_generator, 
      epochs = NUM_EPOCHS,
      verbose = 1,
      validation_data = validation_generator)

# summarize history for accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.xlim([0,NUM_EPOCHS])
plt.ylim([0.4,1.0])
plt.show()
