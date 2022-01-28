import cv2
import numpy as np
import random as rd
import os
import tensorflow as tf


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


def DatasetCreate(abspath, size):
    for i in range(1, 11):
        SourcePath = abspath + '\\{}.jpg'.format(i)
        RetanglePath = abspath + '\\{}\\'.format(i)
        FolderPath = abspath + "\\{}".format(i)
        if not os.path.exists(FolderPath):
            os.makedirs(FolderPath)
        if not os.path.exists(abspath + "\\test"):
            os.makedirs(abspath + "\\test")
        temp = cv2.imread(SourcePath)
        for j in range(0, 361):
            b = RotateBound(temp, j)
            b = cv2.resize(b, (size, size))
            cv2.imshow('title', b)
            cv2.waitKey(1)
            if j >= 300:
                res = abspath + "\\test\\" + str(i) + '-' + str(j) + '.jpg'
                cv2.imwrite(res, b)
            else:
                res = RetanglePath + str(i) + '-' + str(j) + '.jpg'
                cv2.imwrite(res, b)


def ImgToNumpy(image, theory=cv2.COLOR_BGR2RGB, size=None):
    image = cv2.imread(image)
    if size:
        image = cv2.resize(image, size)
    image = cv2.cvtColor(image, theory)
    return image


def TrainSetLoad(abspath):
    TempArray = []
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for i in range(1, 11):
        for j in range(0, 300):
            path = abspath + '\\' + str(i) + "\\" + str(i) + "-" + str(j) + ".jpg"
            temp = ImgToNumpy(path)
            TempArray.append([temp, i])
    TestLength = len(TempArray)
    TempArray = np.array(TempArray)
    np.random.shuffle(TempArray)
    for i in range(TestLength):
        x_train.append(TempArray[i][0])
        y_train.append(TempArray[i][1])
        if i > TestLength * 0.9:
            x_test.append(TempArray[i][0])
            y_test.append(TempArray[i][1])
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    return (x_train, y_train), TestDataLoad(abspath)


def RandomTest(TestCases, x_test, y_test, model):
    right = 0
    wrong = 0
    wrongarr = []
    wrongpic = [0] * 11
    for i in range(TestCases):
        l = rd.randint(0, 350)
        a = model.predict(x_test[l:l + 4])
        for j in a:
            PredictIndex = int(tf.argmax(j))
            ActucalIndex = int(tf.argmax(y_test[l]))
            l += 1
            if PredictIndex != ActucalIndex:
                wrong += 1
                wrongarr.append([PredictIndex, ActucalIndex])
                wrongpic[ActucalIndex] += 1
            else:
                right += 1
    return right, wrong, wrongarr, np.array(wrongpic)


def TestDataLoad(abspath):
    dirpath = abspath + "\\test\\"
    listdir = os.listdir(dirpath)
    temp = []
    x_test = []
    y_test = []
    for i in listdir:
        path = abspath + "\\test\\" + i
        index = i.index("-")
        temp.append([ImgToNumpy(path), int(i[0:index])])
    np.random.shuffle(temp)
    for i in temp:
        x_test.append(i[0])
        y_test.append(i[1])
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return x_test, y_test


def CameraTest(Testcases):
    x_test = []
    y_test = []
    cap = cv2.VideoCapture(0)
    for i in range(Testcases):
        temp = input("Press any key to screenshot")
        ret, frame = cap.read()
        cv2.imshow('pic', frame)
        cv2.waitKey(2000)
        frame = cv2.cvtColor(frame, cv2.cv2.COLOR_BGR2RGB)
        x_test.append(frame)
        y_test.append(int(input("set the testdata label")))
        cv2.destroyAllWindows()
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return x_test, y_test
