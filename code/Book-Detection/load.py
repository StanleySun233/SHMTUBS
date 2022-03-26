import tensorflow as tf

import function as sf

(x_train, y_train), (x_test, y_test) = sf.TrainSetLoad()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

x_test = x_test.astype('float32') / 255.0
y_test = y_test.astype('int8')
y_test_label = list(y_test.copy())

y_test = tf.keras.utils.to_categorical(y_test)

model = tf.keras.models.load_model('./model32.h5')
model.summary()
right, wrong = sf.RandomTest(100, x_test, y_test_label, model)
print(right / (right + wrong))
