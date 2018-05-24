import sympy
import cougr.frameworks.tensorflow


tf_example_filename = 'cougr/frameworks/example_graphs/tensorflow/tf_example_graph.meta'

def run_tf_calculate_tests():
    graph = cougr.frameworks.tensorflow.import_graph(tf_example_filename)
    assert graph.isValid()

    algorithmic_flops = graph.calcAlgFlops()

    add_dim_0 = sympy.Symbol('add:0::dim_0')
    matmul_dim_0 = sympy.Symbol('matmul:0::dim_0')
    mul_dim_0 = sympy.Symbol('mul:0::dim_0')
    correct_alg_flops = 256 * add_dim_0 + \
                        65536 * matmul_dim_0 + \
                        256 * mul_dim_0 + \
                        98307

    assert sympy.simplify(algorithmic_flops - correct_alg_flops) == 0, \
        'Initial alg flops incorrect!\n  Expecting: {}\n  Calculated: {}' \
        .format(correct_alg_flops, algorithmic_flops)

    # Now, bind tensor names in the graph and verify that the algorithmic
    # Flop counts reflect the new name bindings
    batch_size = sympy.Symbol('batch_size')
    # NOTE: This also works: batch_size = 'batch_size'
    # Bind placeholders (a and b) output dimensions 0 to name batch_size
    bind_dict = { 'a': (0, batch_size),
                  'b': (0, batch_size) }
    graph.bindTensorShapeDimensions(bind_dict)

    algorithmic_flops = graph.calcAlgFlops()

    # Update the algorithmic Flops formula
    correct_alg_flops = correct_alg_flops.subs({ add_dim_0: batch_size,
                                                 matmul_dim_0: batch_size,
                                                 mul_dim_0: batch_size, })

    assert sympy.simplify(algorithmic_flops - correct_alg_flops) == 0, \
        'Bound alg flops incorrect!\n  Expecting: {}\n  Calculated: {}' \
        .format(correct_alg_flops, algorithmic_flops)


if __name__ == "__main__":
    run_tf_calculate_tests()

