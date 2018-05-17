
import cougr.frameworks.tensorflow


tf_example_filename = 'cougr/frameworks/example_graphs/tensorflow/tf_example_graph.meta'

def run_tf_calculate_tests():
    graph = cougr.frameworks.tensorflow.import_graph(tf_example_filename)
    assert graph.isValid()

    algorithmic_flops = graph.calcAlgFlops()

    import sympy
    dim_0 = sympy.Symbol('dim_0')
    correct_alg_flops = 65536 * dim_0 + 131074

    assert sympy.simplify(algorithmic_flops - correct_alg_flops) == 0, \
        'Counted algorithmic flops {}'.format(algorithmic_flops)


if __name__ == "__main__":
    run_tf_import_test()

