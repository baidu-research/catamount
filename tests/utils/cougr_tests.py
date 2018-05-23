import sympy

import cougr
from cougr.graph import Graph
from cougr.ops import *
from cougr.tensors import *


def placeholder(graph, name, out_shape):
    ph_op = PlaceholderOp(name)
    out_tensor = Tensor(name, TensorShape(out_shape))
    ph_op.addOutput(out_tensor)
    graph.addOp(ph_op)
    return out_tensor


def variable(graph, name, out_shape):
    var_op = VariableOp(name)
    out_tensor = Tensor(name, TensorShape(out_shape))
    var_op.addOutput(out_tensor)
    graph.addOp(var_op)
    return out_tensor


def matmul(graph, name, out_shape, in_a, in_b):
    mm_op = MatMulOp(name)
    out_tensor = Tensor(name, TensorShape(out_shape))
    mm_op.addOutput(out_tensor)
    graph.addOp(mm_op)
    graph.addInputToOp(mm_op, in_a)
    graph.addInputToOp(mm_op, in_b)
    return out_tensor


def pointwise(graph, name, op_type, out_shape, in_a, in_b):
    op = op_type(name)
    out_tensor = Tensor(name, TensorShape(out_shape))
    op.addOutput(out_tensor)
    graph.addOp(op)
    graph.addInputToOp(op, in_a)
    graph.addInputToOp(op, in_b)
    return out_tensor


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

    # General flow for creating CouGr graph:
    # A) Create graph
    # B) Create op
    # C) Create op output tensor, then addOutput to op
    # D) Tell graph to connect op's inputs to prior op output tensors

    # Model definition parts:
    # 0) Create graph
    graph = Graph()

    # 1) Embedding layer
    # 2) Recurrent layers
    # 3) Projection layer
    lstm_seq = placeholder(graph, 'lstm_out', [batch_size, hidden_dim])
    proj_weights = variable(graph, 'projection_weights', [hidden_dim, projection_dim])
    proj_seq = matmul(graph, 'projection', [batch_size, projection_dim], lstm_seq, proj_weights)

    # 4) Output layer
    output_weights = variable(graph, 'output_weights', [projection_dim, vocab_size])
    output = matmul(graph, 'output_projection', [batch_size, vocab_size], proj_seq, output_weights)
    output_bias = variable(graph, 'output_bias', [vocab_size])
    output = pointwise(graph, 'output_point', AddOp, [batch_size, vocab_size], output, output_bias)

    assert graph.isValid()

    algorithmic_flops = graph.calcAlgFlops()

    # Expected algorithmic Flops
    l_o_dim_0 = sympy.Symbol('lstm_out::dim_0')
    l_o_dim_1 = sympy.Symbol('lstm_out::dim_1')
    p_dim_0 = sympy.Symbol('projection::dim_0')
    p_dim_1 = sympy.Symbol('projection::dim_1')
    p_w_dim_0 = sympy.Symbol('projection_weights::dim_0')
    p_w_dim_1 = sympy.Symbol('projection_weights::dim_1')
    o_w_dim_0 = sympy.Symbol('output_weights::dim_0')
    o_w_dim_1 = sympy.Symbol('output_weights::dim_1')
    o_p_dim_0 = sympy.Symbol('output_projection::dim_0')
    o_p_dim_1 = sympy.Symbol('output_projection::dim_1')
    o_o_dim_0 = sympy.Symbol('output_point::dim_0')
    o_o_dim_1 = sympy.Symbol('output_point::dim_1')
    correct_alg_flops = 2 * l_o_dim_1 * p_dim_0 * p_dim_1 + \
                        o_o_dim_0 * o_o_dim_1 + \
                        2 * o_p_dim_0 * o_p_dim_1 * p_dim_1

    subs = {}
    if batch_size is not None:
        subs[p_dim_0] = batch_size
        subs[o_p_dim_0] = batch_size
    if vocab_size is not None:
        subs[o_w_dim_1] = vocab_size
        subs[o_p_dim_1] = vocab_size
    if hidden_dim is not None:
        subs[l_o_dim_1] = hidden_dim
        subs[p_w_dim_0] = hidden_dim
    if projection_dim is not None:
        subs[p_dim_1] = projection_dim
        subs[o_w_dim_0] = projection_dim
    correct_alg_flops = correct_alg_flops.subs(subs)

    assert sympy.simplify(algorithmic_flops - correct_alg_flops) == 0, \
        'Bound alg flops incorrect!\n  Expecting: {}\n  Calculated: {}' \
        .format(correct_alg_flops, algorithmic_flops)


if __name__ == "__main__":
    run_manual_graph_test()


