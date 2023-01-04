import scipy.io as scio
import numpy as np

ORL_filename = 'data/ORLData_25.mat'
vehicle_filename = 'data/vehicle.mat'

def dataloader(dataset='orl'):
    if dataset == 'orl':
        original_data = scio.loadmat(ORL_filename)
        data = original_data['ORLData']
        data = np.array(data)
        image = np.transpose(data[: -1, :]).astype('float64')
        label = data[-1, :]
        return np.array(image), np.array(label)
    elif dataset == 'vehicle':
        original_data = scio.loadmat(vehicle_filename)
        data = original_data['UCI_tenbin_data']['train_data'][0, 0]
        data = np.array(data)
        image = np.transpose(data[: -1, :]).astype('float64')
        label = data[-1, :]
        return np.array(image), np.array(label)

def data_spliter(data, lable):
    train_data = []
    test_data = []
    train_lable = []
    test_lable = []
    for l in list(set(lable)):
        X_lable = data[np.where(lable == l)]
        # 80%训练，20%测试
        split = int(len(X_lable) * 0.8)
        X_train = X_lable[:split, :]
        X_test = X_lable[split:, :]
        Y_train = [l] * len(X_train)
        Y_test = [l] * len(X_test)
        for x in X_train:
            train_data.append(x)
        for x in X_test:
            test_data.append(x)
        for y in Y_train:
            train_lable.append(y)
        for y in Y_test:
            test_lable.append(y)
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    train_lable = np.array(train_lable)
    test_lable = np.array(test_lable)

    return train_data, train_lable, test_data, test_lable




