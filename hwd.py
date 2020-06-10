from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dropout


(x_train, train_labels),(x_test,test_labels) = mnist.load_data()

img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train/255.0
x_test = x_test/255.0

number_of_classes = 10
train_labels = tf.keras.utils.to_categorical(train_labels, number_of_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, number_of_classes)


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])
# model.fit(x_train, train_labels,
          # batch_size=128,
          # epochs=15,
          # validation_data=(x_test, test_labels))

# model.evaluate(x_test,test_labels)


# model.save('/Users/a./Desktop/course/mnist Model')

from tensorflow import keras
model2 = keras.models.load_model('/Users/a./Desktop/course/mnist Model')

from PIL import Image
import numpy as np
from skimage import transform
def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')
   np_image = transform.resize(np_image, (28, 28, 1))
   np_image = np.expand_dims(np_image, axis=0)
   np_image=np_image/255.0
   return np_image

image = load('/Users/a./Downloads/1272_2280_bundle_archive/testSample/img_47.jpg')
print(model2.predict_classes(image))
