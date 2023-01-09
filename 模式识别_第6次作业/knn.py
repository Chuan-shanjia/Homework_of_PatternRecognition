from sklearn.cluster import KMeans
import numpy as np
from numpy.linalg import cholesky
import random
import matplotlib.pyplot as plt

def get_data():
    sampleNo = 200

    mu1 = np.array([[1, -1]])
    mu2 = np.array([[5.5, -4.5]])
    mu3 = np.array([[1, 4]])
    mu4 = np.array([[6, 4.5]])
    mu5 = np.array([[9, 0]])

    Sigma = np.array([[1, 0], [0, 1]])
    R = cholesky(Sigma).T
    va,vc = np.linalg.eig(Sigma); R2 = (np.diag(va)**0.5)@vc.T

    s1 = np.random.randn(sampleNo, 2) @ R + mu1
    s2 = np.random.randn(sampleNo, 2) @ R + mu2
    s3 = np.random.randn(sampleNo, 2) @ R + mu3
    s4 = np.random.randn(sampleNo, 2) @ R + mu4
    s5 = np.random.randn(sampleNo, 2) @ R + mu5
    s = np.vstack((s1,s2,s3,s4,s5))
    real_mean_vector = [mu1,mu2,mu3,mu4,mu5]

    return s,real_mean_vector

    plt.plot(*s1.T,'.',label = 's1')
    plt.plot(*s2.T,'.',label = 's2')
    plt.plot(*s3.T,'.',label = 's3')
    plt.plot(*s4.T,'.',label = 's4')
    plt.plot(*s5.T,'.',label = 's5')
    plt.axis('scaled')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    melons,real_mean_vector = get_data()
    kmeans = KMeans(n_clusters=2, random_state=0)

    k = 5
    rnd = 0
    ROUND_LIMIT = 20
    THRESHOLD = 1e-10
    clusters = []
    mean_vectors = [[1,1],[1.3,1.5],[1.5,1.6],[1.5,1.3],[1.6,1.5]]

    while True:
        rnd += 1
        change = 0
        clusters = []
        for i in range(k):
            clusters.append([])
        for melon in melons:
            c = np.argmin(
                list(map(lambda vec: np.linalg.norm(melon - vec, ord=2), mean_vectors))
            )

            clusters[c].append(melon)

        for i in range(k):

            new_vector = np.zeros((1, 2))
            for melon in clusters[i]:
                new_vector += melon
            new_vector /= len(clusters[i])

            change += np.linalg.norm(mean_vectors[i] - new_vector, ord=2)
            mean_vectors[i] = new_vector

        if rnd > ROUND_LIMIT or change < THRESHOLD:
            break

    print('最终迭代%d轮' % rnd)

    colors = ['red', 'green', 'blue', 'black', 'yellow']

    for i, col in zip(range(k), colors):
        for melon in clusters[i]:
            plt.scatter(melon[0], melon[1], color=col)

    plt.show()
    print(real_mean_vector)
    print(mean_vectors)