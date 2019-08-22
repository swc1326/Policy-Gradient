import numpy as np
import pandas as pd

def one_hot(n):
    one_hot_ = np.zeros((n, n))
    for same in range(n):
        one_hot_[same][same] = 1
    return one_hot_
def get_train_test_data(all_data_len,train_i,type):
    path = str('./data/'+str(type)+'_'+str(train_i)+'.csv')
    price = pd.read_csv(path)
    data_i = list(price['data'])
    data = {'Data': data_i}
    X =  np.array(data['Data'])
    return X  # numpy
def data_reshape_to_batch_form(X,round_size):
    temp_x = []
    for i in range(round_size,len(X)):
        temp_x.append(X[i-round_size:i])
    return temp_x
def data_prepro(n_actions, n_state, all_data_len,train_i,type,round_size):
    one_hot_state = one_hot(n_state)
    one_hot_action = one_hot(n_actions)
    X = get_train_test_data(all_data_len,train_i,type)
    X = data_reshape_to_batch_form(X,round_size)  ####lstm
    return one_hot_state,one_hot_action,X

