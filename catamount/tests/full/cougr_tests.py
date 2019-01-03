import sympy

import catamount

from catamount.api import utils
from catamount.tests.utils.helpers import *


def softmax(name, out_shape, input, axis=1):
    output = pointwise('{}/exp'.format(name), catamount.ExpOp, out_shape, input)
    reduce_shape = [out_shape[1 - axis]]
    reduced = reduce('{}/reduce'.format(name), 'Sum', reduce_shape,
                     output, axes=axis)
    normd_out = pointwise('{}/div'.format(name), catamount.DivOp, out_shape,
                          output, reduced)
    return normd_out


def linear(name, weights_shape, out_shape, input):
    output_weights = variable('{}_weights'.format(name), weights_shape)
    output = matmul('{}_projection'.format(name), out_shape, input,
                    output_weights)
    output_bias = variable('{}_bias'.format(name), [out_shape[1]])
    output = pointwise('{}_point'.format(name), catamount.AddOp, out_shape,
                       output, output_bias)
    return output


def lstm_cell(name, input, state):

    batch_size = input.shape.dims[0]
    hidden_dim = input.shape.dims[1]
    if hidden_dim.value is None:
        hidden_dim = None

    if hidden_dim is not None:
        in_dim = 2 * hidden_dim
        out_dim = 4 * hidden_dim
    else:
        in_dim = None
        out_dim = None

    assert len(state) == 2
    c, h = state
    lstm_concat_seq = concat('{}_concat'.format(name), [batch_size, in_dim], [h, input], axis=1)
    recur_linear = linear('{}_proj'.format(name), [in_dim, out_dim], [batch_size, out_dim], lstm_concat_seq)
    i, j, f, o = split('{}_split'.format(name), [batch_size, hidden_dim], recur_linear, num_split=4, axis=1)
    forget_bias = variable('{}_f_bias'.format(name), [hidden_dim])
    i = pointwise('{}_i_sig'.format(name), catamount.SigmoidOp, [batch_size, hidden_dim], i)
    j = pointwise('{}_j_tanh'.format(name), catamount.TanhOp, [batch_size, hidden_dim], j)
    f = pointwise('{}_f_add'.format(name), catamount.AddOp, [batch_size, hidden_dim], f, forget_bias)
    f = pointwise('{}_f_sig'.format(name), catamount.SigmoidOp, [batch_size, hidden_dim], f)
    o = pointwise('{}_o_sig'.format(name), catamount.SigmoidOp, [batch_size, hidden_dim], o)
    mul_i_j = pointwise('{}_i_j_mul'.format(name), catamount.MulOp, [batch_size, hidden_dim], i, j)
    new_c = pointwise('{}_c_mul'.format(name), catamount.MulOp, [batch_size, hidden_dim], c, f)
    new_c = pointwise('{}_c_add'.format(name), catamount.AddOp, [batch_size, hidden_dim], new_c, mul_i_j)
    new_c_sig = pointwise('{}_new_c_tanh'.format(name), catamount.TanhOp, [batch_size, hidden_dim], new_c)
    new_h = pointwise('{}_new_h'.format(name), catamount.MulOp, [batch_size, hidden_dim], new_c_sig, o)
    state = [new_c, new_h]
    return new_h, state


def test_manual_graph_build():
    ''' Manually constructs a Catamount graph for a simplified word-level LSTM
    as described in Jozefowicz et al., Exploring the Limits of Language
    Modeling (here: https://arxiv.org/pdf/1602.02410.pdf).

    In particular, implement the following functionality:
    1) input_seq = placeholders(<batch_size * seq_len, vocab_size>)
       embedding_weights = variable(<vocab_size, hidden_dim>)
       lstm_seq = one_hot_lookup(embedding_weights, input_seq)
    2) for layer_id in range(num_layers):
           recur_input = placeholder(<batch_size, hidden_dim>)
           lstm_layer_weights = variable(<2*hidden_dim, 4*hidden_dim>)
           lstm_seq = lstm_layer(lstm_layer_weights, lstm_seq, recur_input)
    3) projection_weights = variable(<hidden_dim, proj_dim>)
       proj_seq = linear(projection_weights, lstm_seq)
    4) output_weights = variable(<proj_dim, vocab_size>)
       outputs = linear(output_weights, proj_seq)
       outputs = softmax(outputs)

    NOTE: linear() is MatMul + BiasAdd
    '''

    # Sizes of everything
    batch_size_str = 'batch_size'
    seq_length_str = 'seq_length'
    vocab_size_str = 'vocab_size'
    hidden_dim_str = 'hidden_dim'
    num_layers_str = 'num_layers' # TODO: Can we make this show up in output?
    projection_dim_str = 'projection_dim'

    batch_size = None
    seq_length = None
    vocab_size = None
    hidden_dim = None
    num_layers = 1
    projection_dim = None

    # Model definition parts:
    # 0) Create graph
    graph = catamount.get_default_graph()

    # 1) Embedding layer
    input_seq = placeholder('input', [batch_size, vocab_size])
    lstm_seq = input_seq

    # 2) Recurrent layers
    for layer_id in range(num_layers):
        layer_name = 'lstm_layer_{}'.format(layer_id)
        # print('Instantiating recurrent layer {}: {}'
        #       .format(layer_id, layer_name))

        # [_] TODO (Joel): Make this recurrent!
        c_state = variable('{}_c_state'.format(layer_name), [batch_size, hidden_dim])
        h_state = variable('{}_h_state'.format(layer_name), [batch_size, hidden_dim])
        # [_] TODO: Would like it to look like this...
        # counter = 0 (constant)
        # new_state = [c_state, h_state]
        # while counter < seq_length: # The condition
        #     # The body
        #     lstm_seq, new_state = lstm_cell(layer_name, lstm_seq, new_state)
        lstm_seq, new_state = lstm_cell(layer_name, lstm_seq, [c_state, h_state])

    # 3) Projection layer
    proj_weights = variable('projection_weights',
                            [hidden_dim, projection_dim])
    proj_seq = matmul('projection', [batch_size, projection_dim],
                      lstm_seq, proj_weights)

    # 4) Output layer
    output = linear('output', [projection_dim, vocab_size],
                    [batch_size, vocab_size], proj_seq)
    normd_out = softmax('output_softmax', [batch_size, vocab_size],
                        output)

    assert graph.isValid()

    algorithmic_flops = graph.calcAlgFlops()

    # Expected algorithmic Flops
    correct_alg_flops = get_correct_alg_flops()
    subs_table = get_subs_table()
    correct_alg_flops = correct_alg_flops.subs(subs_table)

    print('Catamount:   {}'.format(algorithmic_flops))
    print('Correct: {}'.format(correct_alg_flops))
    assert sympy.simplify(algorithmic_flops - correct_alg_flops) == 0, \
        'Bound alg flops incorrect!\n  Expecting: {}\n  Calculated: {}' \
        .format(correct_alg_flops, algorithmic_flops)

    feed_dict = { 'input': [batch_size_str, vocab_size_str],
                  'projection_weights': [hidden_dim_str, projection_dim_str],
                  'output_weights': [projection_dim_str, vocab_size_str],
                  'output_bias': [vocab_size_str] }

    for idx in range(num_layers):
        lstm_layer_name = 'lstm_layer_{}'.format(idx)
        feed_dict['{}_f_bias'.format(lstm_layer_name)] = [hidden_dim_str]
        feed_dict['{}_c_state'.format(lstm_layer_name)] = \
            [batch_size_str, hidden_dim_str]
        feed_dict['{}_h_state'.format(lstm_layer_name)] = \
            [batch_size_str, hidden_dim_str]
    graph.bindShapesAndPropagate(feed_dict)

    assert graph.isValid()

    print(graph.calcAlgFlops())

    print(graph)


if __name__ == "__main__":
    test_manual_graph_build()


