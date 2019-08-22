import tensorflow as tf
def initial(n_actions,n_state,learning_rate,test_data_len,model_i):
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
    tvars = tf.trainable_variables()

    r= tf.placeholder(tf.float32, [None, n_actions], name="input_r")
    constant = tf.placeholder(tf.float32, name="Constant")
    same_state_sum = -tf.reduce_sum(r * constant * probability, axis=1)
    loss=tf.reduce_sum(same_state_sum,axis=0)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        saver.save(sess, './Fully_PG/model_' + str(model_i) + '/fully_initial.ckpt')
