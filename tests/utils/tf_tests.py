
import cougr.frameworks.tensorflow


tf_example_filename = 'cougr/frameworks/example_graphs/tensorflow/tf_example_graph.meta'

def run_tf_calculate_tests():
    graph = cougr.frameworks.tensorflow.import_graph(tf_example_filename)
    assert graph.isValid()

    algorithmic_flops = graph.calcAlgFlops()

    import sympy
    add_dim_1 = sympy.Symbol('add:0::dim_1')
    matmul_dim_0 = sympy.Symbol('matmul:0::dim_0')
    mul_dim_1 = sympy.Symbol('mul:0::dim_1')
    correct_alg_flops = 256 * add_dim_1 + \
                        65536 * matmul_dim_0 + \
                        256 * mul_dim_1 + \
                        98306

    assert sympy.simplify(algorithmic_flops - correct_alg_flops) == 0, \
        'Counted algorithmic flops {}'.format(algorithmic_flops)


if __name__ == "__main__":
    run_tf_import_test()

