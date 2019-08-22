import numpy as np
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
def GR(state_arg, n_actions ,qos_level):
    reward = np.zeros(n_actions)
    for i in range(n_actions):
        reward[i] = -1
    if(qos_level==2.5):
        reward[state_arg] = 1
    elif(qos_level==5):
        if (state_arg != (0)):
            reward[state_arg] = 1
            reward[state_arg -1] = 2
        else:
            reward[state_arg] = 2
    elif(qos_level==7.5):
        if(state_arg != (0) and state_arg != (1)):
            reward[state_arg] = 0#1
            reward[state_arg - 1] = 0#2
            reward[state_arg - 2] = 3
        else:
            if (state_arg == (1)):
                reward[state_arg] = 0#2
                reward[state_arg - 1] = 3
            else:
                reward[state_arg] = 3
    return reward
def GA(max_prob_index,n_actions ,qos_level):
    values = np.zeros(n_actions)
    for i in range(n_actions):
        values[i] = 15- i*2.5
    return values[max_prob_index]
def classify_state(loss_package,qos_level):

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
            elif (7.5 < loss_package <=10):
                state_n = 9
            elif (10 < loss_package <= 12.5):
                state_n = 10
            elif (12.5 < loss_package):
                state_n = 11
        return state_n

def train(batch_size,WS, n_actions, n_state, learning_rate, epoch, one_hot_state, one_hot_action, X,  model_i,qos_level):
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
    constant= tf.placeholder(tf.float32, name="Constant")
    same_state_sum = -tf.reduce_sum(r *constant * probability, axis=1)
    loss=tf.reduce_sum(same_state_sum, axis=0)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    restore_path = './Fully_PG/model_' + str(model_i) + '/fully_initial.ckpt'
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, restore_path)
        down_count = len(X)
        epoch_loss_list = np.zeros(epoch)
        for c in range(epoch):
            print('epoch = ',c)
            G = 10
            G_list = []
            X_list = []
            batch_reward, batch_state, all_reward_for_loss,all_state_for_loss = [], [], [], []
            count = 0
            for x in range(len(X)):
                if(G>80):
                    G_list.append(80)
                    G=80
                else:
                    G_list.append(G)
                loss_package = G-X[x]
                X_list.append(X[x])
                state_arg = classify_state(loss_package,qos_level)
                reward = GR(state_arg, n_actions,qos_level)
                batch_reward.append(reward)
                all_reward_for_loss.append(reward)
                state = one_hot_state[state_arg]
                batch_state.append(state)
                all_state_for_loss.append(state)
                state = np.reshape(state,[1,n_state])
                tfprob = sess.run(probability, feed_dict={input: state})
                max_prob_index = np.argmax(tfprob[0])
                action_values=GA(max_prob_index,n_actions,qos_level)

                G = G + action_values
                with open('./Fully_PG/model_' + str(model_i) + '/train_state_predict.csv',
                          'a',
                          newline='') as p:
                    writer = csv.writer(p)
                    writer.writerow([c,state_arg,max_prob_index,G-action_values])  # 計算每個state的reward正負個數
                if((x+1) % batch_size == 0 ):
                    count = count+1
                    batch_state = np.reshape(batch_state, [batch_size, n_state])
                    batch_reward = np.reshape(batch_reward, [batch_size, n_actions])
                    sess.run(train_op, feed_dict={input: batch_state, r: batch_reward,
                                                  constant: (1 / (down_count))})
                    batch_reward, batch_state = [], []

            epoch_all_loss = np.reshape(all_state_for_loss[0:batch_size*count], [batch_size*count, n_state])
            epoch_all_loss_reward = np.reshape(all_reward_for_loss[0:batch_size * count], [batch_size * count, n_actions])
            epoch_Loss = sess.run(loss, feed_dict={input: epoch_all_loss, r: epoch_all_loss_reward,
                                                   constant: (1 / batch_size*count)})
            epoch_loss_list[c] = epoch_Loss
            if (c % 5 == 0):
                saver.save(sess,
                           './Fully_PG/model_' + str(model_i) + '/' + str(c) + '_fully.ckpt')
                plt.step(np.array(range(len(G_list))), G_list[0:], color='blue', dashes=[6, 2], marker="x")
                # plt.plot(np.array(range(len(G_list))), X, color='red', marker="_")
                plt.step(np.array(range(len(G_list))), X_list[0:], color='red', dashes=[6, 2], marker="x")
                plt.savefig('./Fully_PG/model_' + str(model_i) + '/' + str(c) + '1' + 'draw_train.png')
                plt.close()

        Loss = sess.run(loss, feed_dict={input: epoch_all_loss, r: epoch_all_loss_reward,
                                         constant: (1 / batch_size*count)})

    return Loss, epoch_loss_list