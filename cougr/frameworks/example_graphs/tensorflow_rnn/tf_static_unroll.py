import numpy as np
import os
import tensorflow as tf


def main():
    # Graph:
    batch_size = None
    hidden_dim = 24
    seq_length = 5
    a_dims = [seq_length, batch_size, hidden_dim]
    a = tf.placeholder(tf.float32, shape=a_dims, name='a')
    rnn = tf.contrib.rnn.BasicRNNCell(hidden_dim)
    init_state = tf.placeholder(tf.float32, [batch_size, rnn.state_size],
                                name='init_state')
    curr_state = init_state
    output = []
    for i in range(seq_length):
        out_a, curr_state = rnn(a[i], curr_state)
        output.append(out_a)
    output = tf.stack(output, axis=0, name='stack')

    # print('rnn state_size: {}'.format(rnn.state_size), flush=True)
    # print('init state shape: {}'.format(init_state.shape), flush=True)

    # Session run to get example graph
    batch_size = 8
    init_state_shape = [batch_size, rnn.state_size]
    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          'output_static_unroll')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        a_size = [seq_length, batch_size, hidden_dim]
        feed_dict = {a: np.random.normal(size=a_size),
                     init_state: np.random.normal(size=init_state_shape)}
        out_vals = sess.run([output], feed_dict=feed_dict)
        tf.summary.FileWriter(os.path.join(outdir), sess.graph)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(outdir, 'tf_graph'))


if __name__ == "__main__":
    main()

