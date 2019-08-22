import numpy as np
import pandas as pd

def one_hot(n):
    one_hot_ = np.zeros((n, n))
    for same in range(n):
        one_hot_[same][same] = 1
    return one_hot_
def get_train_test_data(file,type):
    path = str('./data/'+str(type)+'_'+str(file)+'.csv')
    price = pd.read_csv(path)
    data_i = list(price['data'])
    data = {'Data': data_i}
    X =  np.array(data['Data'])
    return X  # numpy
def data_reshape_to_batch_form(X,round_size):
    temp_x = []
    for i in range(round_size,len(X)+1):
        temp_x.append(X[i-round_size:i])
    return temp_x
def lstm_data_prepro(n_actions, n_state,file,type,round_size):
    one_hot_state = one_hot(n_state)
    one_hot_action = one_hot(n_actions)
    X = get_train_test_data(file,type)
    X = data_reshape_to_batch_form(X,round_size)  ####lstm
    return one_hot_state,one_hot_action,X

