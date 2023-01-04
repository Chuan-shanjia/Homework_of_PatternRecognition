import numpy as np


if __name__ == "__main__":
    #读入
    x1 = np.loadtxt('data1.txt')
    x2 = np.loadtxt('data2.txt')
    x3 = np.loadtxt('data3.txt')
    x4 = np.loadtxt('data4.txt')

    x1_train = x1[0:8]
    x2_train = x2[0:8]
    x3_train = x3[0:8]
    x4_train = x4[0:8]
    x_train = np.concatenate((x1_train, x2_train, x3_train, x4_train),axis=0)

    x1_test = x1[8:]
    x2_test = x2[8:]
    x3_test = x3[8:]
    x4_test = x4[8:]
    x_test = np.concatenate((x1_test, x2_test, x3_test, x4_test), axis=0)

    label_train = np.zeros([4, 32])
    label_test = np.zeros([4, 8])

    for i in range(len(label_train)):
        for j in range(len(label_train[0])):
            label_train[i, j] = int(int(j/8)==i)
    for i in range(len(label_test)):
        for j in range(len(label_test[0])):
            label_test[i, j] = int(int(j/2)==i)

    a = np.matmul(np.linalg.pinv(x_train), label_train.T)
    a = a.T
    t = np.arange(1,5)
    test = np.matmul(t, label_test)
    result = np.argmax(np.matmul(a, x_test.T), axis=0) + np.ones_like(np.argmax(np.matmul(a, x_test.T), axis=0))
    correct = sum(test == result) / len(label_test[0])
    print(correct)
