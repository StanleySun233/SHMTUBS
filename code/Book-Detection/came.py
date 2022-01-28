import function as sf
import cv2
import os
import tensorflow as tf

path = 'd:\\py\\book\\tt\\'

dir = os.listdir(path)
img = []
for i in range(len(dir)):
    img.append(sf.ImgToNumpy(path + dir[i]))

print(img)

model = tf.keras.models.load_model('d:\\py\\book\\model64-1.h5')
res = model.predict(img)
print(res)