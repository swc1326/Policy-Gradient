import numpy as np
import tensorflow as tf
import csv

def classify_state(X, n_state):
    up = 80
    if (0 <= X <= 2.5):
        return n_state - 1, 2.5
    for i in range(n_state - 1):
        if (up - (i + 1) * 2.5 < X <= up - i * 2.5):
            return i, up - i * 2.5

def GA(max_prob_index, n_actions):
    values = np.zeros(n_actions)
    jjj = 0
    for i in range(n_actions):
        values[i] = jjj
        jjj = jjj + 2.5
    return values[max_prob_index]


def GR(X, x, n_actions, round_size, n_state):
    values = np.zeros(n_actions)
    jjj = 0
    for i in range(n_actions):
        values[i] = jjj
        jjj = jjj + 2.5
    reward = np.zeros(n_actions)
    flag = 0
    _, down = classify_state(X[x + 1][round_size - 1], n_state)

    for i in range(n_actions):
        if (down + 2.5 >= values[i] > down):
            reward[i] = 1
        elif (down + 5 >= values[i] >= down + 2.5):
            reward[i] = 2
        elif (down + 7.5 >= values[i] > down+5):
            reward[i] = 3
        else:
            reward[i] = -1

    return reward, flag, values


def classify_losspackge(diff, one_hot_state, n_state):
    if (diff == 0):
        class_one_hot = one_hot_state[0]
    for i in range(int((n_state / 2) - 1)):
        if (2.5 * i < diff <= 2.5 * (i + 1)):
            class_one_hot = one_hot_state[i + 1]
    if (2.5 * (int(n_state / 2) - 1) < diff):
        class_one_hot = one_hot_state[int(n_state / 2) - 1]

    for i in range(int(n_state / 2) - 2):
        if (-2.5 * (i + 1) <= diff < -2.5 * (i)):
            class_one_hot = one_hot_state[int(n_state / 2) - 1 + i + 1]
    if (-2.5 * (int(n_state / 2) - 2) > diff):
        class_one_hot = one_hot_state[int(n_state / 2) - 1 + int(n_state / 2) - 2 + 1]
    return class_one_hot


def lstm_test(cell_number,  n_actions, n_state, epoch, one_hot_state,  X,
           model_i,  round_size):
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    input = tf.placeholder(tf.float32, [None, round_size , n_state], name="input_x")  # 1*30
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=cell_number, state_is_tuple=True)
    _, final_state = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=input, dtype=tf.float32)
    W3 = tf.get_variable("W3", shape=[cell_number, n_actions],
                         initializer=tf.contrib.layers.xavier_initializer())
    B3 = tf.get_variable("B3", shape=[1, n_actions],
                         initializer=tf.constant_initializer())
    score = tf.matmul(final_state[1], W3) + B3
    probability = tf.nn.softmax(score)

    restore_path = './model_' + str(model_i) + '/' + str(epoch) + '.ckpt'
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, restore_path)
        down_count = len(X)


        RAB = np.zeros(2)
        DLR = np.zeros(2)
        G_list = []
        X_list = []
        G =80 #15.5
        G_list.append(G)
        batch_reward, batch_state, all_reward_for_loss, batch_action, all_action_for_loss = [], [], [], [], []
        g_count=0
        for x in range(len(X)-1):

            if (x != 0):
                if (G > 80):
                    G_list.append(80)
                else:
                    G_list.append(action_values)

            g_count=g_count+1
            R_state = []
            for i in range(round_size):
                #print(len(X[x][i]))
                state_arg, D = classify_state(X[x][i], n_state)
                state_ = one_hot_state[state_arg]
                R_state.append(state_)

            batch_state.append(R_state)
            state = np.reshape(R_state, [1, round_size , n_state])
            tfprob = sess.run(probability, feed_dict={input: state})
            max_prob_index = np.argmax(tfprob[0])
            loss_package = G - X[x][round_size - 1]
            if (loss_package >= 0):
                RAB[0] = RAB[0] + loss_package
                RAB[1] = RAB[1] + 1
            else:
                DLR[0] = DLR[0] + (-1) * loss_package
                DLR[1] = DLR[1] + 1
            action_values = GA(max_prob_index, n_actions)
            reward, flag, values = GR(X, x, n_actions, round_size, n_state)
            X_list.append(X[x][round_size - 1])
            G = action_values
            batch_reward.append(reward)
            all_reward_for_loss.append(reward)

        x_count=down_count

    if (RAB[1] != 0):
        RAB_ = RAB[0] / RAB[1]
    else:
        RAB_ = 0
    if (DLR[1] != 0):
        DLR_ = DLR[0] / DLR[1]
    else:
        DLR_ = 0
    with open('./model_' + str(model_i) + '/lost_package.csv', 'a',
              newline='') as p:
        writer = csv.writer(p)
        writer.writerow(['RAB', 'DLR'])
        writer.writerow([RAB_, DLR_])

    return x_count,g_count, G_list