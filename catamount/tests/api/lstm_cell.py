import sympy

import catamount
from catamount.graph import Graph
from catamount.tensors.tensor_shape import Dimension

from catamount.tests.utils.helpers import *


# [_] TODO (Joel): Move these to the Catamount API
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
        in_dim = Dimension(2) * hidden_dim
        out_dim = Dimension(4) * hidden_dim
    else:
        in_dim = None
        out_dim = None

    assert len(state) == 2
    c, h = state
    lstm_concat_seq = concat('{}_concat'.format(name), [batch_size, in_dim], [h, input], axis=1)
    recur_linear = linear('{}_proj'.format(name), [in_dim, out_dim], [batch_size, out_dim], lstm_concat_seq)
    i, j, f, o = split('{}_split'.format(name), [batch_size, hidden_dim], recur_linear, axis=1, num_split=4)
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


def test_lstm_cell():
    ''' Specify graph containing an LSTM cell and make sure it generates the
    correct number of Flops. Test combination specifies batch_size and
    hidden_dim.
    '''

    combos = [[None, None],
              [32, None],
              [None, 1024],
              [32, 1024]]

    for combo in combos:
        graph = Graph()
        with graph.asDefault():
            batch_size, hidden_dim = combo
            print('Testing LSTM cell with in batch_size {}, hidden_dim {}'
                  .format(batch_size, hidden_dim))
            input_ph = placeholder('input', [batch_size, hidden_dim])
            state_c_ph = placeholder('c_state', [batch_size, hidden_dim])
            state_h_ph = placeholder('h_state', [batch_size, hidden_dim])
            out_t, state_t = lstm_cell('lstm_cell', input_ph,
                                       [state_c_ph, state_h_ph])

            algorithmic_flops = graph.calcAlgFlops()

            correct_alg_flops = get_correct_alg_flops()
            subs_table = get_subs_table()
            correct_alg_flops = correct_alg_flops.subs(subs_table)
            print('    Catamount:   {}'.format(algorithmic_flops))
            print('    Correct: {}'.format(correct_alg_flops))
            assert sympy.simplify(algorithmic_flops - correct_alg_flops) == 0,\
                'Alg flops incorrect!\n  Expecting:  {}\n  Calculated: {}' \
                .format(correct_alg_flops, algorithmic_flops)
            # TODO: Bind Nones and check outputs
        reset_symbols()


if __name__ == "__main__":
    test_lstm_cell()


