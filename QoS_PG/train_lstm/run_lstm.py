import numpy as np
import matplotlib.pyplot as plt
import os
from data_preprocessing import data_prepro
from initial_model import initial
from Train import train

learning_rate = 0.0005
n_actions=33
n_state=32
round_size = 15 #目標函式中的k值，也是window的size

###可以刪除的###
WS = 1
train_len_data_1 = 200 # data_3 len : 307
test_len_data_1 =60 # data_1 len : 64
all_data_len = train_len_data_1 + WS-1 + round_size -1
test_data_len = test_len_data_1 + WS-1 + round_size -1
###

epoch = 2000
model_num =50
train_i ='average'#'add_rare'#'0_5_9_2_4_6'#'made1''one_to_night'
tend='0'
tend_list = ['0']#['1','2']
qos_level =2.5 #2.5 5 7.5
cell_number =8
add_epoch_to = 1600 #???

for model_i in range(14,model_num):
    path = './model_' + str(model_i)
    isExists = os.path.exists(path)
    if not isExists:
        print(path,'建立成功')
        os.makedirs(path)

    initial(cell_number,n_actions,n_state,learning_rate,model_i,round_size)
    for batch_size_i in range(1, 2):
        batch_size = (2 ** batch_size_i)
        one_hot_state, one_hot_action, X = data_prepro(n_actions, n_state, all_data_len,train_i,'train',round_size)
        loss, epoch_loss_list = train(cell_number,batch_size,n_actions, n_state, learning_rate, epoch, one_hot_state, X, model_i,qos_level,round_size)
        plt.plot(np.array(range(epoch)), epoch_loss_list)
        plt.savefig('./model_' + str(model_i) + '/' + tend + 'loss.png')
        plt.close()
