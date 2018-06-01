import numpy as np
import os
import tensorflow as tf


def main():
    # Graph:
    batch_size = None
    hidden_dim = 1
    a_dims = [batch_size, hidden_dim]
    a = tf.placeholder(tf.float32, shape=a_dims, name='a')
    timer = tf.constant(0, name='timer')
    bound = tf.placeholder(tf.int32, shape=(), name='bound')
    total_inc = tf.get_variable('total_inc', shape=(), dtype=tf.int32)

    def condition(time, *_):
        with tf.variable_scope('cond'):
            return time < bound

    def body(time, input, state):
        with tf.variable_scope('body'):
            total_inc = time
            return (time + 1, input, state + input)

    state = tf.zeros_like(a, name='start_val')
    _, _, state = tf.while_loop(cond=condition,
                                body=body,
                                loop_vars=(timer, a, state))

    # print('rnn state_size: {}'.format(rnn.state_size), flush=True)
    # print('init state shape: {}'.format(init_state.shape), flush=True)

    # Session run to get example graph
    batch_size = 8
    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          'output_simple_while')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        a_dims = [batch_size, hidden_dim]
        feed_dict = {a: np.random.normal(size=a_dims), bound: 25 }
        out_vals = sess.run([state], feed_dict=feed_dict)
        print(out_vals)
        tf.summary.FileWriter(os.path.join(outdir), sess.graph)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(outdir, 'tf_graph'))


if __name__ == "__main__":
    main()

