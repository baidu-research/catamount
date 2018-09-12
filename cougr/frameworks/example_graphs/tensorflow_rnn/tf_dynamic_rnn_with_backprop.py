import numpy as np
import os
import tensorflow as tf


class Optimizer():
    def __init__(self, learning_rate):
        self.opt = tf.train.GradientDescentOptimizer(learning_rate)

    def __call__(self, loss):
        with tf.variable_scope("Gradient"):
            with tf.variable_scope("Compute"):
                gradient = self.opt.compute_gradients(loss)

            with tf.variable_scope("Apply"):
                train_step = self.opt.apply_gradients(gradient)
        return train_step

def main():
    # Graph:
    batch_size = None
    hidden_dim = 24
    seq_length = None
    a_dims = [batch_size, seq_length, hidden_dim]
    out_corr_shape = [batch_size, seq_length]
    a = tf.placeholder(tf.float32, shape=a_dims, name='a')
    rnn = tf.contrib.rnn.BasicLSTMCell(hidden_dim)
    c_init_state = tf.placeholder(tf.float32, [batch_size, hidden_dim],
                                  name='c_init_state')
    h_init_state = tf.placeholder(tf.float32, [batch_size, hidden_dim],
                                  name='h_init_state')
    init_state = tf.contrib.rnn.LSTMStateTuple(c_init_state, h_init_state)
    output, state = tf.nn.dynamic_rnn(rnn, a, initial_state=init_state)
    out_correct = tf.placeholder(tf.int32, shape=out_corr_shape,
                                 name='out_correct')
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=out_correct,
               logits=output, name='softmax')
    opt = Optimizer(0.1)
    train_op = opt(loss)

    # print('rnn state_size: {}'.format(rnn.state_size), flush=True)
    # print('init state shape: {}'.format(init_state.shape), flush=True)
    # print('out correct shape: {}'.format(out_correct.shape), flush=True)

    # Session run to get example graph
    batch_size = 8
    seq_length = 5
    init_state_shape = [batch_size, hidden_dim]
    out_corr_shape = [batch_size, seq_length]
    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          'output_dynamic_rnn_with_backprop')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        a_size = [batch_size, seq_length, hidden_dim]
        init_state_vals = [np.random.normal(size=init_state_shape),
                           np.random.normal(size=init_state_shape)]
        feed_dict = {a: np.random.normal(size=a_size),
                     c_init_state: np.random.normal(size=init_state_shape),
                     h_init_state: np.random.normal(size=init_state_shape),
                     out_correct: np.random.randint(0, 1, size=out_corr_shape)}
        loss_val, _ = sess.run([loss, train_op], feed_dict=feed_dict)
        print('Loss: {}'.format(loss_val))
        tf.summary.FileWriter(os.path.join(outdir), sess.graph)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(outdir, 'tf_graph'))


if __name__ == "__main__":
    main()

