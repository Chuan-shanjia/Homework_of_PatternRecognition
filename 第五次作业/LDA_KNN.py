import os
import numpy as np
import operator
from data_process import dataloader, data_spliter
from pylab import *

class KNN:
    def __init__(self, train_data, train_lables, test_data, test_lables, k=1):
        self.train_data = train_data
        self.test_data = test_data
        self.train_lables = train_lables
        self.test_lables = test_lables
        self.k = k

    def compute_knn_class(self, data):
        diffMat = data - self.train_data
        sqDiffMat = diffMat ** 2
        sqDistinces = sqDiffMat.sum(axis=1)
        distances = sqDistinces ** 0.5
        sortedDistIndicies = distances.argsort()

        # knn
        classCount = {}
        for i in range(self.k):
            voteIlabel = self.train_lables[sortedDistIndicies[i]]
            classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        predicted = sortedClassCount[0][0]
        return predicted

    def compute_accuracy(self):
        count = 0
        for i in range(len(self.test_data)):
            if self.compute_knn_class(self.test_data[i]) == self.test_lables[i]:
                count += 1
        accuracy = count / len(self.test_data)
        return accuracy

class LDA():
    def __init__(self, X, y, k):
        self.X = X
        self.y = y
        self.k = k
        self.W = self.fit()

    def fit(self):
        data = self.X
        label = self.y
        data = data.astype(np.float64)

        data_mean = data - data.mean(axis=0)
        data_mean = data_mean[:, :, np.newaxis]
        # St
        st = np.matmul(data_mean, data_mean.transpose(0, 2, 1)).mean(axis=0)
        # Sw
        unique_arr = np.unique(label)
        sw = np.zeros_like(st)
        xj_mean_list = []
        xj_num_list = []
        for j in unique_arr:
            xj = data_mean[(label == j).squeeze(), :]
            xj_mean_list.append(xj.mean(axis=0))
            xj_num_list.append(xj.shape[0])
            swj = np.matmul(xj, xj.transpose(0, 2, 1)).mean(axis=0)
            sw += swj
        sw /= len(unique_arr)
        # Sb
        sb = np.zeros_like(st)
        for xj_mean, xj_num in zip(xj_mean_list, xj_num_list):
            sb += xj_num * np.matmul(xj_mean, xj_mean.transpose())

        e_vals, e_vecs = np.linalg.eigh(np.linalg.inv(sw).dot(sb))
        idx = np.real(e_vals).argsort()[::-1]
        sorted_e_vecs = e_vecs[:, idx]
        return np.real(sorted_e_vecs)[:, :self.k]

    def reduction(self, data):
        return (np.dot(data, self.W))

def lda_knn(dataset=None, k=10):
    data, lable = dataloader(dataset)
    data_train, lable_train, data_test, lable_test = data_spliter(data, lable)
    lda = LDA(data_train, lable_train, k)
    data_train = lda.reduction(data_train)
    data_test = lda.reduction(data_test)
    knn = KNN(data_train, lable_train, data_test, lable_test, 1)
    acc = knn.compute_accuracy()
    print('LDA+KNN(dataset: {}), \t acc:{:.2}'.format(dataset, acc))
    return acc

def show(data):
    x_axis = range(1, len(data) + 1)
    plt.plot(x_axis, data, 'b.-')
    plt.title('LDA+KNN')
    plt.xlabel('dimension')
    plt.ylabel('accuracy')
    plt.ylim(0, 1.00)
    plt.show()

if __name__ == '__main__':
    acc_orl = []
    acc_vehicle = []
    for k in range(1, 40):
        acc_orl.append(lda_knn('orl', k))
    for k in range(1, 18):
        acc_vehicle.append(lda_knn('vehicle', k))
    show(acc_orl)
    show(acc_vehicle)

