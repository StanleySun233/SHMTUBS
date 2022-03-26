import tkinter as tk
from tkinter.filedialog import (askopenfilename)

import cv2
import numpy as np
import tensorflow.keras as keras
from PIL import Image, ImageTk

import function as sf

global pic

model = keras.models.load_model('./model32.h5')


def pic2TKpic(img, img_size):
    img_ = cv2.imread(img)
    img__ = cv2.resize(img_, img_size)
    img__ = cv2.cvtColor(img__, cv2.COLOR_BGR2RGB)
    img___ = Image.fromarray(img__)
    img____ = ImageTk.PhotoImage(image=img___)
    return img____


def predictTask(*args):
    global pic
    res = askopenfilename()
    Label2.config(text=res)
    pic = pic2TKpic(res, img_size=(200, 200))
    Label3.configure(image=pic)

    f = sf.ImgToNumpy(res, size=32)
    a = model.predict(np.array([f]))
    a = np.argmax(a, axis=1)[0]
    print(a)
    Label5.config(text=str(a))

    # f = open('./dataset/text/{}.txt'.format(a), 'r', encoding='utf-8').readlines()
    # txt = ""
    # for i in f:
    #     txt += i
    # Label6.config(text=txt)


path = ''

mainWindow = tk.Tk()
mainWindow.geometry("800x600")
mainWindow.title('图书分类')
mainWindow.resizable(False, False)

Label1 = tk.Label(mainWindow, text='文件路径:', bg="LightBlue")
Label1.place(x=20, y=20, width=120, height=40)

Label2 = tk.Label(mainWindow, text='暂未选择路径', bg="LightBlue", anchor='w')
Label2.place(x=160, y=20, width=480, height=40)

button1 = tk.Button(mainWindow, text='选择路径', bg='green', command=predictTask)
button1.place(x=660, y=20, width=120, height=40)

Label3 = tk.Label(mainWindow, text='选择图片后显示', bg='LightBlue')
Label3.place(x=20, y=80, width=400, height=400)

Label4 = tk.Label(mainWindow, text="预测标签：", bg='LightBlue')
Label4.place(x=440, y=80, width=80, height=40)

Label5 = tk.Label(mainWindow, text="", bg='Yellow')
Label5.place(x=540, y=80, width=40, height=40)

mainWindow.mainloop()
