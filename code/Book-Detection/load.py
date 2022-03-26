import tensorflow as tf

import function as sf

x_train, y_train, x_test, y_test = sf.TrainSetLoad()

x_test = x_test.astype('float32') / 255.0
y_test = y_test.astype('int8')
y_test_label = y_test

y_test = tf.keras.utils.to_categorical(y_test)

model = tf.keras.models.load_model('E:\\py\\book\\model64-1.h5')
model.summary()

right, wrong = sf.RandomTest(20, x_test, y_test, model)
print(right / (right + wrong))
