import tensorflow as tf
def initial(cell_number,n_actions,n_state,learning_rate,model_i,round_size):
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
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        saver.save(sess, './model_' + str(model_i) + '/initial.ckpt')
