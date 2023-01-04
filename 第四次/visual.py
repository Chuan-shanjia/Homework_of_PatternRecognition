import numpy as np
import matplotlib.pyplot as plt

def visual():
    d = [0.2, 0.02, 0.002, 2]
    datas = []
    for i in d:
        with open(f'./one_record_20_{i}.txt', 'r') as f:
            data = f.readlines()
        data = np.array([j.strip().split() for j in data])
        data = [float(j) for j in data[:, 0]]
        datas.append(data)
    x = np.arange(1, 1501)
    fig, axes = plt.subplots()
    x_kedu = axes.get_xticklabels()
    [i.set_fontname('Times New Roman') for i in x_kedu]
    y_kedu = axes.get_yticklabels()
    [i.set_fontname('Times New Roman') for i in y_kedu]
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
    for i in range(len(datas)):
        plt.plot(x, datas[i], label=f'Learning_rate={d[i]}')
    plt.show()

visual()