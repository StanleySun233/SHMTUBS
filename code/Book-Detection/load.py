import function as sf
import random as rd
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow.keras
import os

(x_test, y_test) = sf.TestDataLoad('E:\\py\\book')

x_test = x_test.astype('float32') / 255.0
y_test = y_test.astype('int8')
print(y_test)
y_test_label = y_test

y_test = tf.keras.utils.to_categorical(y_test)

model = tf.keras.models.load_model('E:\\py\\book\\model64-1.h5')
model.summary()

right, wrong, wrongarr, wrongpic = sf.RandomTest(20, x_test, y_test, model)
print(right / (right + wrong))
