import numpy as np
import matplotlib.pyplot as plt

from mlp_data_preprocessing import mlp_data_prepro
from mlp_test import mlp_test

from lstm_data_preprocessing import lstm_data_prepro
from Lstm_test import lstm_test


learning_rate = 0.0005
n_actions=33
n_state=32
round_size = 15
epoch = 100
model_num =1
file ='compare'#'average'
remainder =2.5
mlp_remainder =7.5
cell_number =8
mlp_c = 495
lstm_c = 537#1999#
mlp_n_actions = 12
mlp_n_state = 12



for model_i in range(model_num):
    for batch_size_i in range(1, 2):
        batch_size = (2 ** batch_size_i)

        one_hot_state, one_hot_action, X = mlp_data_prepro(mlp_n_actions, mlp_n_state, file, 'train')
        f_x,fully_G,fully_X =mlp_test( mlp_n_actions, mlp_n_state, learning_rate, batch_size,  'fully', mlp_c, one_hot_state,
              X, mlp_remainder,round_size)

        one_hot_state, one_hot_action, X = lstm_data_prepro(n_actions, n_state, file, 'train', round_size)
        x_count, g_count,lstm_G = lstm_test(cell_number,n_actions, n_state, lstm_c,
                                      one_hot_state, X, 'lstm',
                                      round_size)
        fully_G.extend(lstm_G)
        plt.step(np.array(range(len(fully_G))), fully_G[0:], color='blue', marker="x")
        plt.step(np.array(range(len(fully_X))), fully_X[0:], color='red', dashes=[6, 2], marker="x")
        plt.savefig('./test.png')
        #plt.close()
        #plt.show()
