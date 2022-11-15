# -*- coding: utf-8 -*-
"""
Created on Thu May  5 00:05:54 2022

@author: Chris
"""

##### turn certificate verification off  #####
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

## import libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
import certifi
import pdb
import numpy as np

from tensorflow.keras import datasets
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model, model_from_json
import matplotlib.image as mpimg 
from keras.utils import plot_model
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
#x_test = x_test / 255.0
x_test = x_test.astype('float32')

#z-score
mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))
x_test = (x_test-mean)/(std+1e-7)
y_test = np_utils.to_categorical(y_test,10)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def load_image(filename):
	img = load_img(filename, target_size=(32, 32))
	img = img_to_array(img)
	img = img.reshape(1, 32, 32, 3)
	img = img / 255.0
	return img


# load the trained CIFAR10 model
MdNamePath='model_5' #the model path
with open(MdNamePath+'.json') as f:
    model = model_from_json(f.read());
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['accuracy'])        
model.load_weights(MdNamePath+'.hdf5');
model.summary()
plot_model(model,to_file='model.png',show_shapes=True)
model5 = mpimg.imread('model.png')
plt.imshow(model5)
plt.axis('off')
#plt.show()
plt.close()
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=1)
print("\nTest set accuracy:  ", test_acc*100.0, "%\n")
#pdb.set_trace()
# get the image from the internet
URL = "https://wagznwhiskerz.com/wp-content/uploads/2017/10/home-cat.jpg"
picture_path  = tf.keras.utils.get_file(origin=URL)
img = load_image(picture_path)
result = model.predict(img)

# show the picture
image = plt.imread(picture_path)
plt.imshow(image)

# show prediction result.
print('\nPrediction: This image most likely belongs to ' + class_names[int(result.argmax(axis=-1))])

####
# get the image from the internet
URL = "https://img.freepik.com/free-photo/airplane-flying-cloudy-sky_144627-132.jpg"
picture_path  = tf.keras.utils.get_file(origin=URL)
img = load_image(picture_path)
result = model.predict(img)

# show the picture
image = plt.imread(picture_path)
plt.imshow(image)

# show prediction result.
print('\nPrediction: This image most likely belongs to ' + class_names[int(result.argmax(axis=-1))])

####
# get the image from the internet
URL = "https://img.freepik.com/free-vector/modern-blue-urban-adventure-suv-vehicle-illustration_1344-205.jpg"
picture_path  = tf.keras.utils.get_file(origin=URL)
img = load_image(picture_path)
result = model.predict(img)

# show the picture
image = plt.imread(picture_path)
plt.imshow(image)

# show prediction result.
print('\nPrediction: This image most likely belongs to ' + class_names[int(result.argmax(axis=-1))])

####
# get the image from the internet
URL = "https://thumbs.dreamstime.com/z/cruise-ship-1753236.jpg"
picture_path  = tf.keras.utils.get_file(origin=URL)
img = load_image(picture_path)
result = model.predict(img)

# show the picture
image = plt.imread(picture_path)
plt.imshow(image)

# show prediction result.
print('\nPrediction: This image most likely belongs to ' + class_names[int(result.argmax(axis=-1))])


####
# get the image from the internet
URL = "https://www2.illinois.gov/dnr/education/WAImages/WAWMWhitetailedDeer.JPG"
picture_path  = tf.keras.utils.get_file(origin=URL)
img = load_image(picture_path)
result = model.predict(img)

# show the picture
image = plt.imread(picture_path)
plt.imshow(image)

# show prediction result.
print('\nPrediction: This image most likely belongs to ' + class_names[int(result.argmax(axis=-1))])

####
