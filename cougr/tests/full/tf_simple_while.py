import sympy
import cougr.frameworks.tensorflow


tf_example_filename = 'cougr/frameworks/example_graphs/tensorflow_rnn/output_simple_while/tf_graph.meta'

def test_tf_simple_while_loop():
    graph = cougr.frameworks.tensorflow.import_graph(tf_example_filename)
    assert graph.isValid()

    algorithmic_flops = graph.calcAlgFlops()

    while_iters = sympy.Symbol('while/LoopCond_block::iters')
    wba_dim_0 = sympy.Symbol('while/body/add_1:0::dim_0')
    correct_alg_flops = while_iters * (wba_dim_0 + 2)

    print('Loaded Flops test:')
    print('    CouGr:   {}'.format(algorithmic_flops))
    print('    Correct: {}'.format(correct_alg_flops))
    assert sympy.simplify(algorithmic_flops - correct_alg_flops) == 0, \
        'Initial alg flops incorrect!\n  Expecting: {}\n  Calculated: {}' \
        .format(correct_alg_flops, algorithmic_flops)

    # Now, bind tensor names in the graph and verify that the algorithmic
    # Flop counts reflect the new name bindings
    batch_size = sympy.Symbol('batch_size')
    # NOTE: This also works: batch_size = 'batch_size'
    # Bind placeholders (a and b) output dimensions 0 to name batch_size
    bind_dict = { 'a': [batch_size, 1] }
    graph.bindTensorShapeDimensions(bind_dict)

    algorithmic_flops = graph.calcAlgFlops()

    # Update the algorithmic Flops formula
    correct_alg_flops = correct_alg_flops.subs({ wba_dim_0: batch_size })
    print('Bound Flops test:')
    print('    CouGr:   {}'.format(algorithmic_flops))
    print('    Correct: {}'.format(correct_alg_flops))
    assert sympy.simplify(algorithmic_flops - correct_alg_flops) == 0, \
        'Bound alg flops incorrect!\n  Expecting: {}\n  Calculated: {}' \
        .format(correct_alg_flops, algorithmic_flops)


if __name__ == "__main__":
    test_tf_simple_while_loop()

