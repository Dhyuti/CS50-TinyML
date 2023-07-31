# Start with a simple neural network for MNIST
# Note that there are 2 layers, one with 20 neurons, and one with 10
# The 10-neuron layer is our final layer because we have 10 classes we want to classify
# Train this, and you should see it get about 98% accuracy

# Load libraries
import sys

import tensorflow as tf

# This script requires TensorFlow 2 and Python 3
if sys.version_info.major < 3:
    raise Exception((f"The script is developed and tested for Python 3. "
                     f"Current version: {sys.version_info.major}"))

if tf.__version__.split('.')[0] != '2':
    raise Exception((f"The script is developed and tested for tensorflow 2. "
                     f"Current version: {tf.__version__}"))

data = tf.keras.datasets.mnist

(training_images, training_labels), (val_images, val_labels) = data.load_data()

training_images  = training_images / 255.0
val_images = val_images / 255.0
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
                                    tf.keras.layers.Dense(20, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=20, validation_data=(val_images, val_labels))

# Examine the test data
# Using model.evaluate, you can get metrics for a test set 
# In this case we only have a training set and a validation set, so we can try it out with the validation set
# The accuracy will be slightly lower, at maybe 96%
# This is because the model hasn't previously seen this data and may not be fully generalized for all data. Still it's a pretty good score.
# You can also predict images, and compare against their actual label. The [0] image in the set is a number 7, and here you can see that neuron 7 has a 9.9e-1 (99%+) probability, so it got it right!


model.evaluate(val_images, val_labels)

classifications = model.predict(val_images)
print(classifications[0])
print(val_labels[0])

# Modify to inspect learned values
# This code is identical, except that the layers are named prior to adding to the sequential. This allows us to inspect their learned parameters later

data = tf.keras.datasets.mnist

(training_images, training_labels), (val_images, val_labels) = data.load_data()

training_images  = training_images / 255.0
val_images = val_images / 255.0
layer_1 = tf.keras.layers.Dense(20, activation=tf.nn.relu)
layer_2 = tf.keras.layers.Dense(10, activation=tf.nn.softmax)
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
                                    layer_1,
                                    layer_2])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=20)

model.evaluate(val_images, val_labels)

classifications = model.predict(val_images)
print(classifications[0])
print(val_labels[0])

# Inspect weights
print(layer_1.get_weights()[0].size)  
# If you print layer_1.get_weights(), you'll see a lot of data. Let's unpack it. First, there are 2 arrays in the result, so let's look at the first one

print(layer_1.get_weights()[1].size)
# The above code will give you 20 -- the get_weights()[1] contains the biases for each of the 20 neurons in this layer

# Inspecting layer 2
# Printing the get_weights will give us 2 lists, the first a list of weights for the 10 neurons, and the second a list of biases for the 10 neurons
print(layer_2.get_weights()[0].size)

print(layer_2.get_weights()[1].size)
