from matplotlib import pyplot as plt

data = open('trainLoss-CNN.txt', 'r').readlines()

loss = []
acc = []
val_acc = []

for i in data[0].replace("\n", "").split(" "):
    print(i)
    loss.append(float(i))

for i in data[1].replace("\n", "").split(" "):
    acc.append(float(i))

for i in data[2].replace("\n", "").split(" "):
    val_acc.append(float(i))

plt.plot(range(10), loss, label="loss", color='lightblue')
plt.plot(range(10), acc, label="accuracy", color='pink')
plt.plot(range(10), val_acc, label='validate accuracy', color='lightgreen')
plt.legend()
plt.show()
