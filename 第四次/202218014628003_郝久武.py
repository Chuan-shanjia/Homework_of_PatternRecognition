
''' Author: UCAS-Jiuwu Hao '''
''' Data: 2022-11-13 '''
import numpy as np
import random
from math import ceil
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# 数据集：三类样本，每类样本10个数据，数据维度是3
d1 = [[ 1.58, 2.32, -5.8], [ 0.67, 1.58, -4.78], [ 1.04, 1.01, -3.63], [-1.49, 2.18, -3.39], [-0.41, 1.21, -4.73], [1.39, 3.16, 2.87],[ 1.20, 1.40, -1.89], [-0.92, 1.44, -3.22], [ 0.45, 1.33, -4.38],[-0.76, 0.84, -1.96]]
d2 = [[ 0.21, 0.03, -2.21], [ 0.37, 0.28, -1.8], [ 0.18, 1.22, 0.16], [-0.24, 0.93, -1.01], [-1.18, 0.39, -0.39], [0.74, 0.96, -1.16],[-0.38, 1.94, -0.48], [0.02, 0.72, -0.17], [ 0.44, 1.31, -0.14],[ 0.46, 1.49, 0.68]]
d3 = [[-1.54, 1.17, 0.64], [5.41, 3.45, -1.33], [ 1.55, 0.99, 2.69], [1.86, 3.19, 1.51], [1.68, 1.79, -0.87], [3.51, -0.22, -1.39],[1.40, -0.44, -0.92], [0.44, 0.83, 1.97], [ 0.25, 0.68, -0.99],[ 0.66, -0.45, 0.08]]
for i in range(10):
    d1[i].extend([1,1,0,0])
    d2[i].extend([1,0,1,0])
    d3[i].extend([1,0,0,1])
data = d1 + d2 + d3  # 最后一维是1，后面接着label
random.shuffle(data)

class BP_train():
    def __init__(self) -> None:
        self.d = 3            # 扩展前的数据维度
        self.classes = 3      # 类别数
        self.epoches = 1500   # 训练轮数
        self.batch_size = 8   # batch大小
        ''' 批处理方式:batch_size > 1 '''
        ''' 单样本方式:batch_size = 1 '''
        self.lr = 0.1         # 学习率
        self.hidden_num = 20  # 隐含层结点数
        self.loss_f = nn.MSELoss(reduction='sum')  # 损失函数
        self.record = []      # 记录loss和正确率
    def main(self):
        W1 = torch.randn((self.hidden_num,self.d+1),requires_grad=True)
        W2 = torch.randn((self.classes,self.hidden_num),requires_grad=True)
        for epoch in range(self.epoches):
            num = ceil(len(data)/self.batch_size)  
            cor_num = 0
            losses = []
            for b in range(num):
                batch_data = data[b*self.batch_size: min((b+1)*self.batch_size,len(data))]
                batch_x = torch.tensor([i[0:4] for i in batch_data]).T # [4,8]
                batch_label = torch.tensor([i[4:] for i in batch_data],dtype=torch.float32).T
                y = torch.mm(W1,batch_x)  # [20,8]
                y = torch.tanh(y)
                z = torch.mm(W2,y) # [3,8]
                z = torch.sigmoid(z)
                loss = self.loss_f(z,batch_label)/2 
                losses.append(loss.data)
                loss.backward()
                W2.data -= self.lr*W2.grad.data  
                W1.data -= self.lr*W1.grad.data
                W2.grad.zero_() 
                W1.grad.zero_()
                output = torch.argmax(z,dim=0)
                ground = torch.argmax(batch_label,dim=0)
                cor_num += torch.sum(output ==ground)
            epoch_loss = sum(losses)
            corr_rate = cor_num/len(data)
            self.record.append([epoch_loss,corr_rate])
        np.savetxt(f'./one_record_{self.hidden_num}_{self.lr}.txt',self.record,fmt='%s')

    def visual(self):
        d = [0.1,0.01,0.001]
        datas = []
        for i in d:
            with open(f'./record_20_{i}.txt','r') as f:
                data = f.readlines()
            data = np.array([j.strip().split() for j in data])
            data = [float(j) for j in data[:,0]]
            datas.append(data)
        x = np.arange(1,1501)
        fig,axes = plt.subplots()
        x_kedu = axes.get_xticklabels()
        [i.set_fontname('Times New Roman') for i in x_kedu]
        y_kedu = axes.get_yticklabels()
        [i.set_fontname('Times New Roman') for i in y_kedu]
        font1 = {'family':'Times New Roman','weight':'normal','size':20}
        for i in range(len(datas)):
            plt.plot(x,datas[i],label = f'Learning_rate={d[i]}')
        plt.xlabel('Epoch',font = font1)
        plt.ylabel('Loss',font = font1)
        plt.tick_params(labelsize = 20)
        plt.legend(prop = font1)
        plt.show()
if __name__ == '__main__':
    a = BP_train()
    a.main()
