import os
import random
import random as rd

import cv2
import numpy as np


def RotateBound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH))


def DatasetCreate(size, rate=0.7):
    if not os.path.exists('./data/sheet'):
        os.makedirs('./data/sheet')
    if not os.path.exists('./data/sheet/test'):
        os.makedirs('./data/sheet/test')

    for i in range(0, 10):
        SourcePath = './data/origin/{}.jpg'.format(i)
        if not os.path.exists('./data/sheet/{}'.format(i)):
            os.makedirs('./data/sheet/{}'.format(i))
        pic = cv2.imread(SourcePath)
        for j in range(0, 361):
            ret = RotateBound(pic, j)
            ret = cv2.resize(ret, (size, size))
            seed = random.random()
            if seed > rate:
                cv2.imwrite('./data/sheet/test/{}-{}.jpg'.format(i, j), ret)
            else:
                cv2.imwrite("./data/sheet/{}/{}-{}.jpg".format(i, i, j), ret)


def ImgToNumpy(image, theory=cv2.COLOR_BGR2RGB, size=None):
    image = cv2.imread(image)
    if size:
        image = cv2.resize(image, size)
    image = cv2.cvtColor(image, theory)
    return image


def TrainSetLoad():
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for i in range(0, 10):
        for j in os.listdir('./data/sheet/{}/'.format(i)):
            pic = ImgToNumpy('./data/sheet/{}/{}'.format(i, j))
            lab = i
            x_train.append(pic)
            x_test.append(lab)
    for i in os.listdir('./data/sheet/test/'):
        pic = ImgToNumpy('./data/sheet/test/{}'.format(i))
        lab = int(i[0:i.index("-")])
        y_train.append(pic)
        y_test.append(lab)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return (x_train, y_train), (x_test, y_test)


def RandomTest(TestCases, x_test, y_test, model):
    right = 0
    wrong = 0
    for i in range(TestCases):
        l = rd.randint(0, 310)
        print(x_test[l])
        a = model.predict([x_test[l]])
        if int(a) == y_test[l]:
            right += 1
        else:
            wrong += 1
    return right, wrong
