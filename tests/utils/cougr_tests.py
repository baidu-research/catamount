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

def placeholder(graph, name, out_shape):
    add_symbols(name, out_shape)

    ph_op = PlaceholderOp(name)
    out_tensor = Tensor(name, TensorShape(out_shape))
    ph_op.addOutput(out_tensor)
    graph.addOp(ph_op)
    return out_tensor


def variable(graph, name, out_shape):
    add_symbols(name, out_shape)

    var_op = VariableOp(name)
    out_tensor = Tensor(name, TensorShape(out_shape))
    var_op.addOutput(out_tensor)
    graph.addOp(var_op)
    return out_tensor


def matmul(graph, name, out_shape, in_a, in_b):
    add_symbols(name, out_shape)
    global correct_alg_flops
    in_a_dim_1 = '{}::dim_1'.format(in_a.name)
    out_dim_0 = '{}::dim_0'.format(name)
    out_dim_1 = '{}::dim_1'.format(name)
    correct_alg_flops += 2 * symbol_table[in_a_dim_1] * \
        symbol_table[out_dim_0] * symbol_table[out_dim_1]

    mm_op = MatMulOp(name)
    out_tensor = Tensor(name, TensorShape(out_shape))
    mm_op.addOutput(out_tensor)
    graph.addOp(mm_op)
    graph.addInputToOp(mm_op, in_a)
    graph.addInputToOp(mm_op, in_b)
    return out_tensor


def pointwise(graph, name, op_type, out_shape, in_a, in_b=None):
    add_symbols(name, out_shape)
    global correct_alg_flops
    out_dim_0 = '{}::dim_0'.format(name)
    out_dim_1 = '{}::dim_1'.format(name)
    correct_alg_flops += symbol_table[out_dim_0] * symbol_table[out_dim_1]

    op = op_type(name)
    out_tensor = Tensor(name, TensorShape(out_shape))
    op.addOutput(out_tensor)
    graph.addOp(op)
    graph.addInputToOp(op, in_a)
    if in_b is not None:
        graph.addInputToOp(op, in_b)
    return out_tensor


def reduce(graph, name, op_func, out_shape, input, axis=0):
    assert len(input.shape.dims) == 2, \
           'Reduce only supports 2 dimensional input for now'
    assert len(out_shape) == 1, \
           'Reduce only supports 2->1 dimensions for now'
    add_symbols(name, out_shape)
    global correct_alg_flops
    in_dim = '{}::dim_{}'.format(input.name, axis)
    out_dim = '{}::dim_{}'.format(name, 1 - axis)
    correct_alg_flops += symbol_table[in_dim] * symbol_table[out_dim]

    op = ReduceOp(name, axis=axis)
    out_tensor = Tensor(name, TensorShape(out_shape))
    op.addOutput(out_tensor)
    graph.addOp(op)
    graph.addInputToOp(op, input)
    return out_tensor


def softmax(graph, name, out_shape, input, axis=1):
    output = pointwise(graph, '{}/exp'.format(name), ExpOp, out_shape, input)
    reduce_shape = [out_shape[1 - axis]]
    reduced = reduce(graph, '{}/reduce'.format(name), 'Sum', reduce_shape,
                     output, axis=axis)
    normd_out = pointwise(graph, '{}/div'.format(name), DivOp, out_shape,
                          output, reduced)
    return normd_out


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
    graph = Graph()

    # 1) Embedding layer
    input_seq = placeholder(graph, 'input', [batch_size, hidden_dim])

    # 2) Recurrent layers
    for layer_id in range(num_layers):
        print('Instantiating recurrent layer {}: {}'
              .format(layer_id, layer_name))

    # 3) Projection layer
    proj_weights = variable(graph, 'projection_weights',
                            [hidden_dim, projection_dim])
    proj_seq = matmul(graph, 'projection', [batch_size, projection_dim],
                      input_seq, proj_weights)

    # 4) Output layer
    output_weights = variable(graph, 'output_weights',
                              [projection_dim, vocab_size])
    output = matmul(graph, 'output_projection', [batch_size, vocab_size],
                    proj_seq, output_weights)
    output_bias = variable(graph, 'output_bias', [vocab_size])
    output = pointwise(graph, 'output_point', AddOp, [batch_size, vocab_size],
                       output, output_bias)
    normd_out = softmax(graph, 'output_softmax', [batch_size, vocab_size],
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

# TODO (Joel): Add binding tests
#    batch_size = 'batch_size'
#    feed_dict = { 'lstm_out': (0, batch_size) }
#    graph.bind__(feed_dict)
#    print(graph.calcAlgFlops())


if __name__ == "__main__":
    run_manual_graph_test()


