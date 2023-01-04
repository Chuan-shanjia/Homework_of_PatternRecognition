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

class PCA:
    def __init__(self, data, k):
        self.data = data
        self.k = k
        self.W = self.fit()

    # SVD
    def fit(self):
        data = self.data - self.data.mean(axis=0)
        X_cov = np.cov(data.T, ddof=0)
        U, sigma, V = np.linalg.svd(X_cov)
        return U[:, :self.k]

    # 降维
    def reduction(self, data):
        return np.matmul(data, self.W)

def pca_knn(dataset=None, k=10):
    data, lable = dataloader(dataset)
    train_data, train_lable, test_data, test_lable = data_spliter(data, lable)

    pca = PCA(train_data, k)
    train_data = pca.reduction(train_data)
    test_data = pca.reduction(test_data)

    knn = KNN(train_data, train_lable, test_data, test_lable, 1)
    acc = knn.compute_accuracy()
    print('PCA+KNN(dataset: {}), \t acc:{:.2}'.format(dataset, acc))
    return acc

if __name__ == '__main__':
    acc_orl = []
    acc_vehicle = []

    for k in range(1, 30):
        acc_orl.append(pca_knn('orl', k))
        acc_vehicle.append(pca_knn('vehicle', k))

    x_axis = range(1, len(acc_orl) + 1)
    plt.plot(x_axis, acc_orl, 'r.-')
    plt.plot(x_axis, acc_vehicle, 'b.-')
    plt.title('PCA+KNN')
    plt.xlabel('dimension')
    plt.ylabel('accuracy')
    plt.ylim(0, 1.00)
    plt.show()
