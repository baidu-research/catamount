import sympy
import catamount.frameworks.tensorflow
from catamount.api import utils


tf_example_filename = 'catamount/frameworks/example_graphs/tensorflow/rnn/output_static_unroll/tf_graph.meta'

def test_tf_static_unroll_rnn():
    graph = catamount.frameworks.tensorflow.import_graph(tf_example_filename)
    assert graph.isValid()

    algorithmic_flops = graph.calcAlgFlops()

    ba_0 = utils.getIntSymbolFromString('basic_rnn_cell/BiasAdd:0::dim_0')
    ba_1 = utils.getIntSymbolFromString('basic_rnn_cell/BiasAdd_1:0::dim_0')
    ba_2 = utils.getIntSymbolFromString('basic_rnn_cell/BiasAdd_2:0::dim_0')
    ba_3 = utils.getIntSymbolFromString('basic_rnn_cell/BiasAdd_3:0::dim_0')
    ba_4 = utils.getIntSymbolFromString('basic_rnn_cell/BiasAdd_4:0::dim_0')
    mm_0 = utils.getIntSymbolFromString('basic_rnn_cell/MatMul:0::dim_0')
    mm_1 = utils.getIntSymbolFromString('basic_rnn_cell/MatMul_1:0::dim_0')
    mm_2 = utils.getIntSymbolFromString('basic_rnn_cell/MatMul_2:0::dim_0')
    mm_3 = utils.getIntSymbolFromString('basic_rnn_cell/MatMul_3:0::dim_0')
    mm_4 = utils.getIntSymbolFromString('basic_rnn_cell/MatMul_4:0::dim_0')
    th_0 = utils.getIntSymbolFromString('basic_rnn_cell/Tanh:0::dim_0')
    th_1 = utils.getIntSymbolFromString('basic_rnn_cell/Tanh_1:0::dim_0')
    th_2 = utils.getIntSymbolFromString('basic_rnn_cell/Tanh_2:0::dim_0')
    th_3 = utils.getIntSymbolFromString('basic_rnn_cell/Tanh_3:0::dim_0')
    th_4 = utils.getIntSymbolFromString('basic_rnn_cell/Tanh_4:0::dim_0')
    correct_alg_flops = 24 * (ba_0 + ba_1 + ba_2 + ba_3 + ba_4) + \
                        2304 * (mm_0 + mm_1 + mm_2 + mm_3 + mm_4) + \
                        144 * (th_0 + th_1 + th_2 + th_3 + th_4)

    print('Loaded Flops test:')
    print('    Catamount:   {}'.format(algorithmic_flops))
    print('    Correct: {}'.format(correct_alg_flops))
    assert sympy.simplify(algorithmic_flops - correct_alg_flops) == 0, \
        'Initial alg flops incorrect!\n  Expecting: {}\n  Calculated: {}' \
        .format(correct_alg_flops, algorithmic_flops)

    # Now, bind tensor names in the graph and verify that the algorithmic
    # Flop counts reflect the new name bindings
    batch_size = utils.getIntSymbolFromString('batch_size')
    # NOTE: This also works: batch_size = 'batch_size'
    # Bind placeholders (a and b) output dimensions 0 to name batch_size
    bind_dict = { 'a': ['seq_length', 'batch_size', 'hidden_dim'],
                  'init_state': ['batch_size', 'hidden_dim'] }
    graph.bindShapesAndPropagate(bind_dict)

    algorithmic_flops = graph.calcAlgFlops()

    # Update the algorithmic Flops formula
    correct_alg_flops = correct_alg_flops.subs({ ba_0: batch_size,
                                                 ba_1: batch_size,
                                                 ba_2: batch_size,
                                                 ba_3: batch_size,
                                                 ba_4: batch_size,
                                                 mm_0: batch_size,
                                                 mm_1: batch_size,
                                                 mm_2: batch_size,
                                                 mm_3: batch_size,
                                                 mm_4: batch_size,
                                                 th_0: batch_size,
                                                 th_1: batch_size,
                                                 th_2: batch_size,
                                                 th_3: batch_size,
                                                 th_4: batch_size })
    print('Bound Flops test:')
    print('    Catamount:   {}'.format(algorithmic_flops))
    print('    Correct: {}'.format(correct_alg_flops))
    assert sympy.simplify(algorithmic_flops - correct_alg_flops) == 0, \
        'Bound alg flops incorrect!\n  Expecting: {}\n  Calculated: {}' \
        .format(correct_alg_flops, algorithmic_flops)


if __name__ == "__main__":
    test_tf_static_unroll_rnn()

