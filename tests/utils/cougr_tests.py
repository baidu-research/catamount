import sympy

import cougr
from cougr.graph import Graph
from cougr.ops import *
from cougr.tensors import *


# Build symbol table with this function (for verifying CouGr
# Flops calculations)
symbol_table = {}
subs_table = {}
correct_alg_flops = 0

def add_symbols(name, out_shape):
    global symbol_table
    global subs_table
    for idx, dim in enumerate(out_shape):
        sym_name = '{}::dim_{}'.format(name, idx)
        symbol = sympy.Symbol(sym_name)
        assert sym_name not in symbol_table.keys()
        symbol_table[sym_name] = symbol
        if dim is not None:
            subs_table[symbol] = dim


# General flow for creating CouGr graph ops:
# a) Create op
# b) Create op output tensor, then addOutput to op
# c) Tell graph to connect op's inputs to prior op output tensors

def placeholder(name, out_shape):
    add_symbols(name, out_shape)
    return cougr.placeholder(name, out_shape)


def variable(name, out_shape):
    add_symbols(name, out_shape)
    return cougr.variable(name, out_shape)


def matmul(name, out_shape, in_a, in_b):
    add_symbols(name, out_shape)
    global correct_alg_flops
    in_a_dim_1 = '{}::dim_1'.format(in_a.name)
    out_dim_0 = '{}::dim_0'.format(name)
    out_dim_1 = '{}::dim_1'.format(name)
    correct_alg_flops += 2 * symbol_table[in_a_dim_1] * \
        symbol_table[out_dim_0] * symbol_table[out_dim_1]
    return cougr.matmul(name, out_shape, in_a, in_b)


def pointwise(name, op_type, out_shape, in_a, in_b=None):
    add_symbols(name, out_shape)
    global correct_alg_flops
    out_dim_0 = '{}::dim_0'.format(name)
    out_dim_1 = '{}::dim_1'.format(name)
    flops_per_elt = 1
    if op_type == SigmoidOp:
        flops_per_elt = 4
    if op_type == TanhOp:
        flops_per_elt = 6
    correct_alg_flops += flops_per_elt * symbol_table[out_dim_0] * \
                         symbol_table[out_dim_1]
    return cougr.pointwise(name, op_type, out_shape, in_a, in_b)


def reduce(name, op_func, out_shape, input, axis=0):
    assert len(input.shape.dims) == 2, \
           'Reduce only supports 2 dimensional input for now'
    assert len(out_shape) == 1, \
           'Reduce only supports 2->1 dimensions for now'
    add_symbols(name, out_shape)
    global correct_alg_flops
    in_dim = '{}::dim_{}'.format(input.name, axis)
    out_dim = '{}::dim_{}'.format(name, 1 - axis)
    correct_alg_flops += symbol_table[in_dim] * symbol_table[out_dim]
    return cougr.reduce(name, op_func, out_shape, input, axis)


def softmax(name, out_shape, input, axis=1):
    output = pointwise('{}/exp'.format(name), ExpOp, out_shape, input)
    reduce_shape = [out_shape[1 - axis]]
    reduced = reduce('{}/reduce'.format(name), 'Sum', reduce_shape,
                     output, axis=axis)
    normd_out = pointwise('{}/div'.format(name), DivOp, out_shape,
                          output, reduced)
    return normd_out


def linear(name, weights_shape, out_shape, input):
    output_weights = variable('{}_weights'.format(name), weights_shape)
    output = matmul('{}_projection'.format(name), out_shape, input,
                    output_weights)
    output_bias = variable('{}_bias'.format(name), [out_shape[1]])
    output = pointwise('{}_point'.format(name), AddOp, out_shape,
                       output, output_bias)
    return output


def split(name, out_shape, input, num_splits=2, axis=0):
    for i in range(num_splits):
        out_name = '{}_out{}'.format(name, i)
        add_symbols(out_name, out_shape)
    return cougr.split(name, out_shape, input, num_splits, axis)


def concat(name, out_shape, input_list, axis=0):
    add_symbols(name, out_shape)
    return cougr.concat(name, out_shape, input_list, axis)


def run_manual_graph_test():
    ''' Manually constructs a CouGr graph for a simplified word-level LSTM
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
    batch_size = None
    seq_length = None
    vocab_size = None
    hidden_dim = None
    num_layers = 2
    projection_dim = None

    # Model definition parts:
    # 0) Create graph
    graph = cougr.get_default_graph()

    # 1) Embedding layer
    input_seq = placeholder('input', [batch_size, vocab_size])
    lstm_seq = input_seq

    # 2) Recurrent layers
    for layer_id in range(num_layers):
        layer_name = 'lstm_layer_{}'.format(layer_id)
        # print('Instantiating recurrent layer {}: {}'
        #       .format(layer_id, layer_name))

        if hidden_dim is not None:
            in_dim = 2 * hidden_dim
            out_dim = 4 * hidden_dim
        else:
            in_dim = None
            out_dim = None

        # [_] TODO (Joel): Wrap this as an LSTM cell. Then, make it recurrent!
        recur_state = variable('{}_init_state'.format(layer_name), [batch_size, in_dim])
        c, h = split('{}_recur_split'.format(layer_name), [batch_size, hidden_dim], recur_state, num_splits=2, axis=1)
        lstm_concat_seq = concat('{}_concat'.format(layer_name), [batch_size, in_dim], [recur_state, lstm_seq], axis=1)
        recur_linear = linear(layer_name, [in_dim, out_dim], [batch_size, out_dim], lstm_concat_seq)
        i, j, f, o = split('{}_split'.format(layer_name), [batch_size, hidden_dim], recur_linear, num_splits=4, axis=1)
        forget_bias = variable('{}_f_bias'.format(layer_name), [hidden_dim])
        f = pointwise('{}_f_add'.format(layer_name), AddOp, [batch_size, hidden_dim], f, forget_bias)
        f = pointwise('{}_f_sig'.format(layer_name), SigmoidOp, [batch_size, hidden_dim], f)
        i = pointwise('{}_i_sig'.format(layer_name), SigmoidOp, [batch_size, hidden_dim], i)
        j = pointwise('{}_j_tanh'.format(layer_name), TanhOp, [batch_size, hidden_dim], j)
        mul_i_j = pointwise('{}_i_j_mul'.format(layer_name), MulOp, [batch_size, hidden_dim], i, j)
        new_c = pointwise('{}_c_mul'.format(layer_name), MulOp, [batch_size, hidden_dim], c, f)
        new_c = pointwise('{}_c_add'.format(layer_name), AddOp, [batch_size, hidden_dim], new_c, mul_i_j)
        o = pointwise('{}_o_sig'.format(layer_name), SigmoidOp, [batch_size, hidden_dim], o)
        new_c_sig = pointwise('{}_new_c_tanh'.format(layer_name), TanhOp, [batch_size, hidden_dim], new_c)
        new_h = pointwise('{}_new_h'.format(layer_name), MulOp, [batch_size, hidden_dim], new_c_sig, o)
        lstm_seq = new_h

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
    global correct_alg_flops
    global subs_table
    correct_alg_flops = correct_alg_flops.subs(subs_table)

    print('CouGr:   {}'.format(algorithmic_flops))
    print('Correct: {}'.format(correct_alg_flops))
    assert sympy.simplify(algorithmic_flops - correct_alg_flops) == 0, \
        'Bound alg flops incorrect!\n  Expecting: {}\n  Calculated: {}' \
        .format(correct_alg_flops, algorithmic_flops)

    batch_size_str = 'batch_size'
    seq_length_str = 'seq_length'
    vocab_size_str = 'vocab_size'
    hidden_dim_str = 'hidden_dim'
    num_layers_str = 'num_layers' # TODO: Can we make this show up in output?
    projection_dim_str = 'projection_dim'
    feed_dict = { 'input': [batch_size_str, vocab_size_str],
                  'projection_weights': [hidden_dim_str, projection_dim_str],
                  'output_weights': [hidden_dim_str, vocab_size_str],
                  'output_bias': [vocab_size_str] }

    for idx in range(num_layers):
        lstm_layer_name = 'lstm_layer_{}'.format(idx)
        feed_dict['{}_f_bias'.format(lstm_layer_name)] = [hidden_dim_str]
        feed_dict['{}_init_state'.format(lstm_layer_name)] = \
            [batch_size_str, None]
    graph.bindTensorShapeDimensions(feed_dict)
    print(graph.calcAlgFlops())


if __name__ == "__main__":
    run_manual_graph_test()


