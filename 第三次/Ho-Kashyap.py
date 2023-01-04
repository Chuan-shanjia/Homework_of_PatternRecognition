import numpy as np


if __name__ == "__main__":
    #读入
    x1 = np.loadtxt('data2.txt')
    x2 = np.loadtxt('data4.txt')
    one = np.ones(10)
    #标准化
    y1 = np.insert(x1, 2, one, 1)
    y2 = np.insert(-x2, 2, -one, 1)

    y = np.append(y1, y2, axis=0)
    print(y)
    print(len(y))
    #训练
    step = 0
    a = np.zeros(3)
    b = np.ones(20)
    lr = 1
    for i in range(1000):
        e = np.subtract(np.matmul(y, a), b)
        e_ = 0.5*(e+np.absolute(e))
        b = b + 2*lr*e_
        a = np.matmul(np.linalg.pinv(y), b)
        if max(abs(e))<0.0001:
            break
    print('a:',a)
    count = 0
    for i in range(len(y)):
        if np.dot(a,y[i]) <= 0:
            count= count+1
    print('training errors:',count)
    print('e:',np.linalg.norm(e))