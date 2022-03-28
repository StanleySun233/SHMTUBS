import tensorflow.keras as keras


class LossHistory(keras.callbacks.Callback):
    def __init__(self, method):
        super().__init__()
        self.losses = None
        self.accuracy = None
        self.method = method

    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('accuracy'))

    def on_train_end(self, logs=None):
        self.write()

    def write(self):
        f = open('trainLoss-{}.txt'.format(self.method), 'w', encoding='utf-8')
        txtLoss = ""
        for i in range(len(self.losses)):
            txtLoss += (str(self.losses[i]) + " ")
        txtAccuracy = ""
        for i in range(len(self.accuracy)):
            txtAccuracy += (str(self.accuracy[i]) + " ")
        txtLoss += '\n'
        txtAccuracy += '\n'
        f.writelines(txtLoss)
        f.writelines(txtAccuracy)
        f.close()
