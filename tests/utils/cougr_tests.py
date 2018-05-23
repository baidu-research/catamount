import sympy

import cougr
from cougr.graph import Graph
from cougr.ops import *
from cougr.tensors import *

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
    lstm_ph = PlaceholderOp('lstm_out')
    lstm_seq = Tensor(lstm_ph.name, TensorShape([batch_size, hidden_dim]))
    lstm_ph.addOutput(lstm_seq)
    graph.addOp(lstm_ph)

    proj_var = VariableOp('projection_weights')
    proj_weights = Tensor(proj_var.name,
                          TensorShape([hidden_dim, projection_dim]))
    proj_var.addOutput(proj_weights)
    graph.addOp(proj_var)

    proj_op = MatMulOp('projection')
    proj_seq = Tensor(proj_op.name, TensorShape([batch_size, projection_dim]))
    proj_op.addOutput(proj_seq)
    graph.addOp(proj_op)
    graph.addInputToOp(proj_op, lstm_seq)
    graph.addInputToOp(proj_op, proj_weights)

    # 4) Output layer
    output_var = VariableOp('output_weights')
    output_weights = Tensor(output_var.name,
                            TensorShape([projection_dim, vocab_size]))
    output_var.addOutput(output_weights)
    graph.addOp(output_var)

    output_op = MatMulOp('output_projection')
    out_tensor = Tensor(output_op.name,
                        TensorShape([batch_size, vocab_size]))
    output_op.addOutput(out_tensor)
    graph.addOp(output_op)
    graph.addInputToOp(output_op, proj_seq)
    graph.addInputToOp(output_op, output_weights)

    assert graph.isValid()

    algorithmic_flops = graph.calcAlgFlops()

    # Expected algorithmic Flops
    o_p_dim_0 = sympy.Symbol('output_projection::dim_0')
    o_p_dim_1 = sympy.Symbol('output_projection::dim_1')
    p_dim_0 = sympy.Symbol('projection::dim_0')
    p_dim_1 = sympy.Symbol('projection::dim_1')
    l_o_dim_1 = sympy.Symbol('lstm_out::dim_1')
    correct_alg_flops = 2 * l_o_dim_1 * p_dim_0 * p_dim_1 + 2 * o_p_dim_0 * o_p_dim_1 * p_dim_1

    assert sympy.simplify(algorithmic_flops - correct_alg_flops) == 0, \
        'Bound alg flops incorrect!\n  Expecting: {}\n  Calculated: {}' \
        .format(correct_alg_flops, algorithmic_flops)


if __name__ == "__main__":
    run_manual_graph_test()


