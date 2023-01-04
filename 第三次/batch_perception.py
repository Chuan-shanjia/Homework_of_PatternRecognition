import numpy as np


if __name__ == "__main__":
    #读入
    x1 = np.loadtxt('data3.txt')
    x2 = np.loadtxt('data2.txt')
    b = np.ones(10)
    #标准化
    y1 = np.insert(x1, 2, b, 1)
    y2 = np.insert(-x2, 2, -b, 1)

    y = np.append(y1, y2, axis=0)
    print(y)
    print(len(y))
    #训练
    step = 0
    a = np.zeros(3)
    lr = 1
    done = False
    while not done:
        Y = []
        for i in range(len(y)):
            b = np.dot(a,y[i])
            if b <= 0.0:
                Y.append(y[i])
        if len(Y) == 0:
            print('a:', a, 'step:', step)
            break
        Y_sum = np.sum(Y, axis=0)
        a = a + lr * Y_sum
        step = step + 1
