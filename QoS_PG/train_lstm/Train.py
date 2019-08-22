import numpy as np
import tensorflow as tf
import csv
import matplotlib.pyplot as plt

def classify_state(X, n_state, qos_level):
    up = 80
    if (0 <= X <= 2.5):
        return n_state - 1, 2.5
    for i in range(n_state - 1):

        if (up - (i + 1) * 2.5 < X <= up - i * 2.5):
            return i, up - i * 2.5
def GA(max_prob_index, n_actions, qos_level):
    values = np.zeros(n_actions)
    jjj = 0
    for i in range(n_actions):
        values[i] = jjj
        jjj = jjj + 2.5
    return values[max_prob_index]

def GR(X, x, n_actions, round_size, qos_level, G, action_values, state_arg, n_state):
    values = np.zeros(n_actions)
    jjj = 0
    for i in range(n_actions):
        values[i] = jjj
        jjj = jjj + 2.5
    reward = np.zeros(n_actions)
    flag = 0
    _, down = classify_state(X[x + 1][round_size - 1], n_state, qos_level)
    for i in range(n_actions):
        if (down + 5 >= values[i] > 2.5 + down):
            reward[i] = 1  # 1
        elif (down + 7.5 >= values[i] >= down + 5):
            reward[i] = 2  # 2
        elif (down + 10 > values[i] > down + 7.5):
            reward[i] = 1  # 3
        else:
            reward[i] = -1
    state = [23, 26, 28, 29, 30]
    state_weight = [20, 6, 15, 20, 12]
    state_index = state.index(state_arg)
    reward = reward * state_weight[state_index]

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

def train(cell_number, batch_size, n_actions, n_state, learning_rate, epoch, one_hot_state, X,
           model_i, qos_level, round_size):
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    input = tf.placeholder(tf.float32, [None, round_size, n_state], name="input_x")  # 1*30
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=cell_number, state_is_tuple=True)
    _, final_state = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=input, dtype=tf.float32)
    W3 = tf.get_variable("W3", shape=[cell_number, n_actions],
                         initializer=tf.contrib.layers.xavier_initializer())
    B3 = tf.get_variable("B3", shape=[1, n_actions],
                         initializer=tf.constant_initializer())
    score = tf.matmul(final_state[1], W3) + B3
    probability = tf.nn.softmax(score)
    r = tf.placeholder(tf.float32, [None, n_actions], name="input_r")
    constant = tf.placeholder(tf.float32, name="Constant")
    same_state_sum = -tf.reduce_sum(r * constant * probability, axis=1)
    loss = tf.reduce_sum(same_state_sum, axis=0)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    restore_path = './model_' + str(model_i) + '/initial.ckpt'
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, restore_path)
        down_count = len(X)
        epoch_loss_list = np.zeros(epoch)
        bad_situation = np.zeros(epoch)
        for c in range(epoch):
            print('epoch= ', c)
            G_list = []
            X_list = []
            G = 15.5
            batch_reward, batch_state, all_reward_for_loss, all_state_for_loss, batch_action, all_action_for_loss = [], [], [], [], [], []
            count = 0
            for x in range(len(X) - 1):
                if (G - X[x][round_size - 1] < 0):
                    bad_situation[c] = bad_situation[c] + 1
                if (x != 0):
                    if (G > 80):
                        G_list.append(80)
                    else:
                        G_list.append(action_values)
                if (x % batch_size == 0 and x != 0):
                    count = count + 1
                    batch_state = np.reshape(batch_state, [batch_size, round_size, n_state])
                    batch_reward = np.reshape(batch_reward, [batch_size, n_actions])
                    sess.run(train_op, feed_dict={input: batch_state, r: batch_reward,
                                                  constant: (1 / (down_count))})
                    batch_reward, batch_action, batch_state = [], [], []

                R_state = []
                for i in range(round_size):
                    state_arg, D = classify_state(X[x][i], n_state, qos_level)
                    state_ = one_hot_state[state_arg]
                    R_state.append(state_)
                batch_state.append(R_state)
                all_state_for_loss.append(R_state)
                state = np.reshape(R_state, [1, round_size, n_state])
                tfprob = sess.run(probability, feed_dict={input: state})
                max_prob_index = np.argmax(tfprob[0])

                state_arg, D = classify_state(X[x][round_size - 1], n_state, qos_level)
                action_values = GA(max_prob_index, n_actions, qos_level)
                reward, flag, values = GR(X, x, n_actions, round_size, qos_level, G, action_values, state_arg,
                                          n_state)
                max_values_index = np.argmax(reward)
                if (c >= 995):
                    with open('./model_' + str(model_i) + '/train_state_predict.csv',
                              'a',
                              newline='') as p:
                        writer = csv.writer(p)
                        writer.writerow(
                            ['epoch', 'G', 'X', 'G - X[x][round_size - 1]', 'state_arg'])
                        writer.writerow(
                            [c, G, X[x][round_size - 1], G - X[x][round_size - 1], state_arg])  # , reward

                if (x != 0):
                    X_list.append(X[x][round_size - 1])

                G = action_values
                batch_reward.append(reward)
                all_reward_for_loss.append(reward)
            epoch_all_loss = np.reshape(all_state_for_loss[:batch_size * count],
                                        [batch_size * count, round_size, n_state])
            epoch_all_loss_reward = np.reshape(all_reward_for_loss[:batch_size * count],
                                               [batch_size * count, n_actions])
            epoch_Loss = sess.run(loss, feed_dict={input: epoch_all_loss, r: epoch_all_loss_reward,
                                                   constant: (1 / batch_size * count)})
            epoch_loss_list[c] = epoch_Loss
            if (bad_situation[c] == 0):
                saver.save(sess,
                           './model_' + str(model_i) + '/' + str(c) + '.ckpt')
                plt.step(np.array(range(len(G_list))), G_list[0:], color='blue', dashes=[6, 2], marker="x")
                plt.step(np.array(range(len(G_list))), X_list[0:], color='red', dashes=[6, 2], marker="x")
                plt.savefig('./model_' + str(model_i) + '/' + str(c) + '.png')
                plt.close()
        check = False
        check_list = []
        for bs in range(len(bad_situation)):
            if (bad_situation[bs] == 0):
                check = True
                check_list.append(bs)
        Loss = sess.run(loss, feed_dict={input: epoch_all_loss, r: epoch_all_loss_reward,
                                         constant: (1 / batch_size * count)})

    return Loss, epoch_loss_list