import sympy
import catamount.frameworks.tensorflow
from catamount.api import utils


tf_example_filename = 'catamount/frameworks/example_graphs/tensorflow/rnn/output_dynamic_rnn/tf_graph.meta'

def test_tf_dynamic_rnn():
    graph = catamount.frameworks.tensorflow.import_graph(tf_example_filename)
    assert graph.isValid()

    algorithmic_flops = graph.calcAlgFlops()

    rwb_iters = utils.getIntSymbolFromString('rnn/while/LoopCond_block::iters')
    ba_0 = utils.getIntSymbolFromString('rnn/while/basic_rnn_cell/BiasAdd:0::dim_0')
    mm_0 = utils.getIntSymbolFromString('rnn/while/basic_rnn_cell/MatMul:0::dim_0')
    th_0 = utils.getIntSymbolFromString('rnn/while/basic_rnn_cell/Tanh:0::dim_0')
    correct_alg_flops = rwb_iters * \
                        (24 * ba_0 + 2304 * mm_0 + 144 * th_0 + 5)

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
    bind_dict = { 'a': ['batch_size', 'seq_length', 'hidden_dim'],
                  'init_state': ['batch_size', 'hidden_dim'] }
    graph.bindShapesAndPropagate(bind_dict)

    algorithmic_flops = graph.calcAlgFlops()

    # Update the algorithmic Flops formula
    correct_alg_flops = correct_alg_flops.subs({ ba_0: batch_size,
                                                 mm_0: batch_size,
                                                 th_0: batch_size })
    print('Bound Flops test:')
    print('    Catamount:   {}'.format(algorithmic_flops))
    print('    Correct: {}'.format(correct_alg_flops))
    assert sympy.simplify(algorithmic_flops - correct_alg_flops) == 0, \
        'Bound alg flops incorrect!\n  Expecting: {}\n  Calculated: {}' \
        .format(correct_alg_flops, algorithmic_flops)


if __name__ == "__main__":
    test_tf_dynamic_rnn()

