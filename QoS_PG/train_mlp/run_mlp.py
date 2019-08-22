import numpy as np
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
import os

from data_preprocessing import data_prepro
from initial_model import initial
from train_data import train

learning_rate = 0.01
n_actions=12 #可能的12種actions
n_state=12 #可能的12種states

WS = 1
train_len_data_1 = 222 # data_3 len : 307
test_len_data_1 = 58 # data_1 len : 64
all_data_len = train_len_data_1 + WS-1
test_data_len = test_len_data_1 + WS-1

epoch = 100
model_num=2
train_i = 'average'
tend='1'
tend_list = ['1']
qos_level =5 

for model_i in range(1, model_num):
    path = './Fully_PG/model_' + str(model_i) #路徑
    isExists = os.path.exists(path)
    if not isExists:
        print(path,'建立成功')
        os.makedirs(path) #創建目錄，即如果判斷檔案不存在就直接建一個新的。

    initial(n_actions, n_state, learning_rate, test_data_len, model_i)
    for batch_size_i in range(1, 2):
        batch_size = (2 ** batch_size_i) #batch_size為一次要收集的lists數量

        one_hot_state, one_hot_action, X = data_prepro(n_actions, n_state, all_data_len, train_i, 'train')
        loss, epoch_loss_list = train(batch_size,WS, n_actions, n_state, learning_rate, epoch, one_hot_state, one_hot_action, X, model_i, qos_level)
        plt.plot(np.array(range(epoch)), epoch_loss_list)
        plt.savefig('./Fully_PG/model_' + str(model_i) + '/' + tend + 'loss.png')
        plt.close()
