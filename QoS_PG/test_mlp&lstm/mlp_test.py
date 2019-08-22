import numpy as np
import tensorflow as tf
import csv

def GR(state_arg, n_actions, mlp_remainder):
    reward = np.zeros(n_actions)
    for i in range(n_actions):
        reward[i] = -1
    if(mlp_remainder==2.5):
        reward[state_arg] = 1
    elif(mlp_remainder==5):
        if (state_arg != (0)):
            reward[state_arg] = 1
            reward[state_arg -1] = 2
        else:
            reward[state_arg] = 2
    elif(mlp_remainder==7.5):
        if(state_arg != (0) and state_arg != (1)):
            reward[state_arg] = 0
            reward[state_arg - 1] = 0
            reward[state_arg - 2] = 3
        else:
            if (state_arg == (1)):
                reward[state_arg] = 0
                reward[state_arg - 1] = 3
            else:
                reward[state_arg] = 3
    return reward

def GA(max_prob_index,n_actions ):
    values = np.zeros(n_actions)
    for i in range(n_actions):
        values[i] = 15- i*2.5
    return values[max_prob_index]

def classify_state(loss_package):
        if(loss_package<0):
            if(loss_package<-12.5):
                state_n=0
            elif(-12.5<=loss_package<-10):
                state_n=1
            elif (-10 <= loss_package < -7.5):
                state_n = 2
            elif (-7.5 <= loss_package < -5):
                state_n = 3
            elif (-5 <= loss_package < -2.5):
                state_n = 4
            elif (-2.5 <= loss_package < 0):
                state_n = 5
        else:
            if (0 <= loss_package <= 2.5):
                state_n = 6
            elif (2.5 < loss_package <= 5):
                state_n = 7
            elif (5 < loss_package <= 7.5):
                state_n = 8
            elif (7.5 < loss_package <= 10):
                state_n = 9
            elif (10 < loss_package <= 12.5):
                state_n = 10
            elif (12.5< loss_package):
                state_n = 11
        return state_n

def mlp_test(n_actions, n_state, learning_rate, batch_size,  model_i, c,one_hot_state,  X,mlp_remainder,round_size):
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    input = tf.placeholder(tf.float32, [None, n_state], name="input_x")  # 1*30
    with tf.name_scope('layer_1'):
        W1 = tf.get_variable("W1", shape=[n_state, 24],
                             initializer=tf.contrib.layers.xavier_initializer())
        B1 = tf.get_variable("B1", shape=[1, 24],
                             initializer=tf.constant_initializer())
        layer1 = tf.nn.relu(tf.matmul(input, W1) + B1)
    with tf.name_scope('layer_3'):
        W3 = tf.get_variable("W3", shape=[24, n_actions],
                             initializer=tf.contrib.layers.xavier_initializer())
        B3 = tf.get_variable("B3", shape=[1, n_actions],
                             initializer=tf.constant_initializer())
        score = tf.matmul(layer1, W3) + B3
    probability = tf.nn.softmax(score)
    r = tf.placeholder(tf.float32, [None, n_actions], name="input_r")
    constant = tf.placeholder(tf.float32, name="Constant")
    same_state_sum = -tf.reduce_sum(r * constant * probability, axis=1)
    loss = tf.reduce_sum(same_state_sum, axis=0)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    restore_path = './model_' + str(model_i) + '/'+ str(c) + '_fully.ckpt'
    saver = tf.train.Saver()
    RAB=np.zeros(2)
    DLR=np.zeros(2)
    with tf.Session() as sess:
        saver.restore(sess, restore_path)
        down_count = len(X)
        G =60#5 #
        batch_reward, batch_state = [], []
        count = 0
        G_list = []
        f_x=0
        for x in range(round_size-1):
            f_x=f_x+1
            if(G>80):
                G_list.append(80)
                G=80
            else:
                G_list.append(G)
            loss_package = G - X[x]
            state_arg = classify_state(loss_package)
            reward = GR(state_arg, n_actions,mlp_remainder)
            batch_reward.append(reward)
            state = one_hot_state[state_arg]
            batch_state.append(state)
            state = np.reshape(state, [1, n_state])
            tfprob = sess.run(probability, feed_dict={input: state})
            max_prob_index = np.argmax(tfprob[0])
            action_values = GA(max_prob_index, n_actions)
            if(loss_package>=0):
                RAB[0]=RAB[0]+loss_package
                RAB[1] = RAB[1] +1
            else:
                DLR[0]=DLR[0]+(-1)*loss_package
                DLR[1] = DLR[1] +1
            G = G + action_values
            if ((x + 1) % batch_size == 0):
                count = count + 1
                batch_state = np.reshape(batch_state, [batch_size, n_state])
                batch_reward = np.reshape(batch_reward, [batch_size, n_actions])
                sess.run(train_op, feed_dict={input: batch_state, r: batch_reward,
                                              constant: (1 / (down_count))})
                batch_reward, batch_state = [], []
        if(RAB[1]!=0):
            RAB_=RAB[0]/RAB[1]
        else:
            RAB_=0
        if(DLR[1]!=0):
            DLR_=DLR[0]/DLR[1]
        else:
            DLR_=0
        with open('./model_' + str(model_i) + '/lost_package.csv', 'a',
                  newline='') as p:
            writer = csv.writer(p)
            writer.writerow(['RAB', 'DLR'])
            writer.writerow([RAB_,DLR_])
        return f_x,G_list,X
