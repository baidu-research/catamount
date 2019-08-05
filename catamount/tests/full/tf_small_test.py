import sympy
import catamount.frameworks.tensorflow
from catamount.api import utils


tf_example_filename = 'catamount/frameworks/example_graphs/tensorflow/simple/tf_example_graph.meta'

def test_tf_load_and_calculate():
    graph = catamount.frameworks.tensorflow.import_graph(tf_example_filename)
    assert graph.isValid()

    algorithmic_flops = graph.calcAlgFlops()

    add_dim_0 = utils.getIntSymbolFromString('add:0::dim_0')
    matmul_dim_0 = utils.getIntSymbolFromString('matmul:0::dim_0')
    mul_dim_0 = utils.getIntSymbolFromString('mul:0::dim_0')
    correct_alg_flops = 256 * add_dim_0 + \
                        65536 * matmul_dim_0 + \
                        256 * mul_dim_0 + \
                        1

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
    bind_dict = { 'a': [batch_size, None],
                  'b': [batch_size, None] }
    graph.bindShapesAndPropagate(bind_dict)

    algorithmic_flops = graph.calcAlgFlops()

    # Update the algorithmic Flops formula
    correct_alg_flops = correct_alg_flops.subs({ add_dim_0: batch_size,
                                                 matmul_dim_0: batch_size,
                                                 mul_dim_0: batch_size, })
    print('Bound Flops test:')
    print('    Catamount:   {}'.format(algorithmic_flops))
    print('    Correct: {}'.format(correct_alg_flops))
    assert sympy.simplify(algorithmic_flops - correct_alg_flops) == 0, \
        'Bound alg flops incorrect!\n  Expecting: {}\n  Calculated: {}' \
        .format(correct_alg_flops, algorithmic_flops)


if __name__ == "__main__":
    test_tf_load_and_calculate()

