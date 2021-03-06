import tensorflow as tf

import call
import function as sf

TestBatchSize, TestEpochs = 1, 10
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
num_class = 10

(x_train, y_train), (x_test, y_test) = sf.TrainSetLoad()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

x_train = x_train.astype('float32') / 255.0
y_train = y_train.astype('int8')
x_test = x_test.astype('float32') / 255.0
y_test = y_test.astype('int8')

y_train_label = y_train
y_test_label = y_test
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

c1_1 = tf.keras.layers.Conv2D(64, (8, 8), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
c1_2 = tf.keras.layers.Conv2D(64, (8, 8), activation='relu', kernel_initializer='he_normal', padding='same')(c1_1)
c1_3 = tf.keras.layers.Conv2D(64, (8, 8), activation='relu', kernel_initializer='he_normal', padding='same')(c1_2)
c1_4 = tf.keras.layers.Conv2D(96, (8, 8), activation='relu', kernel_initializer='he_normal', padding='same')(c1_3)
c1_5 = tf.keras.layers.Conv2D(96, (8, 8), activation='relu', kernel_initializer='he_normal', padding='same')(c1_4)
bt1 = tf.keras.layers.BatchNormalization()(c1_5)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(bt1)
dr1 = tf.keras.layers.Dropout(0.5)(p1)

c2_1 = tf.keras.layers.Conv2D(128, (8, 8), activation='relu', kernel_initializer='he_normal', padding='same')(dr1)
c2_2 = tf.keras.layers.Conv2D(128, (8, 8), activation='relu', kernel_initializer='he_normal', padding='same')(c2_1)
c2_3 = tf.keras.layers.Conv2D(128, (8, 8), activation='relu', kernel_initializer='he_normal', padding='same')(c2_2)
c2_4 = tf.keras.layers.Conv2D(128, (8, 8), activation='relu', kernel_initializer='he_normal', padding='same')(c2_3)
c2_5 = tf.keras.layers.Conv2D(128, (8, 8), activation='relu', kernel_initializer='he_normal', padding='same')(c2_4)
bt2 = tf.keras.layers.BatchNormalization()(c2_5)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(bt2)
dr2 = tf.keras.layers.Dropout(0.5)(p2)

c3_1 = tf.keras.layers.Conv2D(128, (8, 8), activation='relu', kernel_initializer='he_normal', padding='same')(dr2)
c3_2 = tf.keras.layers.Conv2D(128, (8, 8), activation='relu', kernel_initializer='he_normal', padding='same')(c3_1)
c3_3 = tf.keras.layers.Conv2D(128, (8, 8), activation='relu', kernel_initializer='he_normal', padding='same')(c3_2)
c3_4 = tf.keras.layers.Conv2D(128, (8, 8), activation='relu', kernel_initializer='he_normal', padding='same')(c3_3)
c3_5 = tf.keras.layers.Conv2D(128, (8, 8), activation='relu', kernel_initializer='he_normal', padding='same')(c3_4)
bt3 = tf.keras.layers.BatchNormalization()(c3_5)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(bt3)
dr3 = tf.keras.layers.Dropout(0.5)(p3)

flatten3 = tf.keras.layers.Flatten()(dr3)

outputs = tf.keras.layers.Dense(num_class, activation='softmax')(flatten3)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(
    optimizer=tf.keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004),
    # lr=0.005
    loss='CategoricalCrossentropy',
    metrics=['accuracy'])
model.summary()

filepath = './ckeckpoi1.h5'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath, monitor='val_accuracy', verbose=0,
    save_best_only=True, save_weights_only=True,
    save_frequency=1)

callback = call.LossHistory("CNN")
# model.fit(x_train, y_train, batch_size=TestBatchSize, epochs=TestEpochs, verbose=1, callbacks=[checkpoint_callback],
#           validation_data=(x_test, y_test), shuffle=True)
model.fit(x_train, y_train, batch_size=TestBatchSize, epochs=TestEpochs, verbose=1, callbacks=[callback],
          validation_data=(x_test, y_test), shuffle=True)

model.save('./model{}.h5'.format(IMG_WIDTH))
