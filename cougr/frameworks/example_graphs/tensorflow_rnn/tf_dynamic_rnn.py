import numpy as np
import os
import tensorflow as tf


def main():
    # Graph:
    batch_size = None
    hidden_dim = 24
    seq_length = None
    a_dims = [batch_size, seq_length, hidden_dim]
    a = tf.placeholder(tf.float32, shape=a_dims, name='a')
    rnn = tf.contrib.rnn.BasicRNNCell(hidden_dim)
    init_state = tf.placeholder(tf.float32, [batch_size, rnn.state_size],
                                name='init_state')
    output, state = tf.nn.dynamic_rnn(rnn, a, initial_state=init_state)

    # print('rnn state_size: {}'.format(rnn.state_size), flush=True)
    # print('init state shape: {}'.format(init_state.shape), flush=True)

    # Session run to get example graph
    batch_size = 8
    seq_length = 5
    init_state_shape = [batch_size, rnn.state_size]
    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          'output_dynamic_rnn')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        a_size = [batch_size, seq_length, hidden_dim]
        feed_dict = {a: np.random.normal(size=a_size),
                     init_state: np.random.normal(size=init_state_shape)}
        out_vals = sess.run([output], feed_dict=feed_dict)
        tf.summary.FileWriter(os.path.join(outdir), sess.graph)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(outdir, 'tf_graph'))


if __name__ == "__main__":
    main()

