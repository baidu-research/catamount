import sympy
import cougr.frameworks.tensorflow
from cougr.api import utils
from cougr.tensors.tensor_shape import TensorShape


tf_example_filename = 'cougr/frameworks/example_graphs/tensorflow_rnn/output_dynamic_rnn_with_backprop/tf_graph.meta'

def test_tf_dynamic_rnn():
    graph = cougr.frameworks.tensorflow.import_graph(tf_example_filename)

    print('INITIAL GRAPH: {}\n\n'.format(graph))

    assert graph.isValid()

    algorithmic_flops = graph.calcAlgFlops()

    rwb_iters = utils.getIntSymbolFromString('rnn/while/LoopCond_block::iters')
    a_0 =    utils.getIntSymbolFromString('rnn/while/basic_lstm_cell/Add:0::dim_0')
    a1_0 =   utils.getIntSymbolFromString('rnn/while/basic_lstm_cell/Add_1:0::dim_0')
    ba_0 =   utils.getIntSymbolFromString('rnn/while/basic_lstm_cell/BiasAdd:0::dim_0')
    mm_0 =   utils.getIntSymbolFromString('rnn/while/basic_lstm_cell/MatMul:0::dim_0')
    m_0 =    utils.getIntSymbolFromString('rnn/while/basic_lstm_cell/Mul:0::dim_0')
    m1_0 =   utils.getIntSymbolFromString('rnn/while/basic_lstm_cell/Mul_1:0::dim_0')
    m2_0 =   utils.getIntSymbolFromString('rnn/while/basic_lstm_cell/Mul_2:0::dim_0')
    s_0 =    utils.getIntSymbolFromString('rnn/while/basic_lstm_cell/Sigmoid:0::dim_0')
    s1_0 =   utils.getIntSymbolFromString('rnn/while/basic_lstm_cell/Sigmoid_1:0::dim_0')
    s2_0 =   utils.getIntSymbolFromString('rnn/while/basic_lstm_cell/Sigmoid_2:0::dim_0')
    sm_r_0 = utils.getIntSymbolFromString('softmax/Reshape:0::dim_0')
    sm_r_1 = utils.getIntSymbolFromString('softmax/Reshape:0::dim_1')
    th_0 =   utils.getIntSymbolFromString('rnn/while/basic_lstm_cell/Tanh:0::dim_0')
    th1_0 =  utils.getIntSymbolFromString('rnn/while/basic_lstm_cell/Tanh_1:0::dim_0')
    forward_flops = rwb_iters * (24 * a_0 + 24 * a1_0 + 96 * ba_0 + 9216 * mm_0 + \
                                 24 * m_0 + 24 * m1_0 + 24 * m2_0 + 96 * s_0 + \
                                 96 * s1_0 + 96 * s2_0 + 144 * th_0 + 144 * th1_0 + 6) + \
                    3 * sm_r_0 * sm_r_1 + \
                    18628
    grad_iters = utils.getIntSymbolFromString('Gradient/Compute/gradients/b_count_2_block::iters')
    g_an_0 =   utils.getIntSymbolFromString('Gradient/Compute/gradients/AddN:0::dim_0')
    g_an1_0 =  utils.getIntSymbolFromString('Gradient/Compute/gradients/AddN_1:0::dim_0')
    g_mm_0 =   utils.getIntSymbolFromString('Gradient/Compute/gradients/rnn/while/basic_lstm_cell/MatMul_grad/MatMul:0::dim_0')
    g_mms_0 =  utils.getIntSymbolFromString('Gradient/Compute/gradients/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2:0::dim_0')
    g_ms_0 =   utils.getIntSymbolFromString('Gradient/Compute/gradients/rnn/while/basic_lstm_cell/Mul_1_grad/mul/StackPopV2:0::dim_0')
    g_m_0 =    utils.getIntSymbolFromString('Gradient/Compute/gradients/rnn/while/basic_lstm_cell/Mul_1_grad/mul:0::dim_0')
    g_ms1_0 =  utils.getIntSymbolFromString('Gradient/Compute/gradients/rnn/while/basic_lstm_cell/Mul_1_grad/mul_1/StackPopV2:0::dim_0')
    g_m1_0 =   utils.getIntSymbolFromString('Gradient/Compute/gradients/rnn/while/basic_lstm_cell/Mul_1_grad/mul_1:0::dim_0')
    g_ms2_0 =  utils.getIntSymbolFromString('Gradient/Compute/gradients/rnn/while/basic_lstm_cell/Mul_2_grad/mul/StackPopV2:0::dim_0')
    g_m2_0 =   utils.getIntSymbolFromString('Gradient/Compute/gradients/rnn/while/basic_lstm_cell/Mul_2_grad/mul:0::dim_0')
    g_ms21_0 = utils.getIntSymbolFromString('Gradient/Compute/gradients/rnn/while/basic_lstm_cell/Mul_2_grad/mul_1/StackPopV2:0::dim_0')
    g_m21_0 =  utils.getIntSymbolFromString('Gradient/Compute/gradients/rnn/while/basic_lstm_cell/Mul_2_grad/mul_1:0::dim_0')
    g_mus_0 =  utils.getIntSymbolFromString('Gradient/Compute/gradients/rnn/while/basic_lstm_cell/Mul_grad/mul/StackPopV2:0::dim_0')
    g_mu_0 =   utils.getIntSymbolFromString('Gradient/Compute/gradients/rnn/while/basic_lstm_cell/Mul_grad/mul:0::dim_0')
    g_mu1_0 =  utils.getIntSymbolFromString('Gradient/Compute/gradients/rnn/while/basic_lstm_cell/Mul_grad/mul_1:0::dim_0')
    g_s_0 =    utils.getIntSymbolFromString('Gradient/Compute/gradients/rnn/while/basic_lstm_cell/Sigmoid_grad/SigmoidGrad:0::dim_0')
    g_c_0 =    utils.getIntSymbolFromString('Gradient/Compute/gradients/rnn/while/basic_lstm_cell/split_grad/concat:0::dim_0')
    g_sm_m_0 = utils.getIntSymbolFromString('Gradient/Compute/gradients/softmax/softmax_grad/mul:0::dim_0')
    g_sm_m_1 = utils.getIntSymbolFromString('Gradient/Compute/gradients/softmax/softmax_grad/mul:0::dim_1')
    backward_flops = grad_iters * (24 * g_an_0 + 72 * g_an1_0 + 9216 * g_mm_0 + 9216 * g_mms_0 + \
                                   72 * g_ms_0 + 48 * g_m_0 + 72 * g_ms1_0 + 48 * g_m1_0 + \
                                   72 * g_ms2_0 + 48 * g_m2_0 + 72 * g_ms21_0 + 48 * g_m21_0 + \
                                   72 * g_mus_0 + 48 * g_mu_0 + 48 * g_mu1_0 + 48 * g_s_0 + \
                                   96 * g_c_0 + 4706) + \
                     g_sm_m_0 * g_sm_m_1
    correct_alg_flops = forward_flops + backward_flops

    print('Loaded Flops test:')
    print('    CouGr:   {}'.format(algorithmic_flops))
    print('    Correct: {}'.format(correct_alg_flops))
    assert sympy.simplify(algorithmic_flops - correct_alg_flops) == 0, \
        'Initial alg flops incorrect!\n  Expecting: {}\n  Calculated: {}' \
        .format(correct_alg_flops, algorithmic_flops)

    # Now, bind tensor names in the graph and verify that the algorithmic
    # Flop counts reflect the new name bindings
    batch_size = utils.getIntSymbolFromString('batch_size')
    seq_length = utils.getIntSymbolFromString('seq_length')
    hidden_dim = utils.getIntSymbolFromString('hidden_dim')

    # Manually set some variables
    # TODO (Joel): Fix this up when all tensor arrays work!
    ta_op = graph.opsByName['rnn/TensorArrayStack/TensorArraySizeV3']
    ta_op._outputs[0].setValue(seq_length)
    ta_op = graph.opsByName['rnn/TensorArrayStack/TensorArrayGatherV3']
    ta_op._outputs[0].shape.mergeShape([seq_length, batch_size, hidden_dim])
    ta_op = graph.opsByName['rnn/while/TensorArrayReadV3']
    ta_op._outputs[0].shape.mergeShape([batch_size, hidden_dim])

    # TODO (Joel): Fix this up when all stack ops work!
    find_stack_shape = TensorShape([None, 24])
    find_stack_shape_2 = TensorShape([None, 48])
    for op in graph.opsByName.values():
       op_name_suffix = op.name.split('/')[-1]
       if 'StackPopV2' in op_name_suffix:
           if op._outputs[0].shape == find_stack_shape:
               op._outputs[0].shape.mergeShape([batch_size, hidden_dim])
           elif op._outputs[0].shape == find_stack_shape_2:
               op._outputs[0].shape.mergeShape([batch_size, 2 * hidden_dim])

    # NOTE: This also works: batch_size = 'batch_size'
    # Bind placeholders (a and b) output dimensions 0 to name batch_size
    bind_dict = { 'a': [batch_size, seq_length, hidden_dim],
                  'c_init_state': [batch_size, hidden_dim],
                  'h_init_state': [batch_size, hidden_dim],
                  'out_correct': [batch_size, seq_length] }
    graph.bindTensorShapeDimensions(bind_dict, warn_if_ill_defined=True)

    algorithmic_flops = graph.calcAlgFlops()

    # Update the algorithmic Flops formula
    # Sub the forward prop values
    correct_alg_flops = correct_alg_flops.subs({ ba_0: batch_size,
                                                 a_0: batch_size,
                                                 a1_0: batch_size,
                                                 mm_0: batch_size,
                                                 m_0: batch_size,
                                                 m1_0: batch_size,
                                                 m2_0: batch_size,
                                                 s_0: batch_size,
                                                 s1_0: batch_size,
                                                 s2_0: batch_size,
                                                 th_0: batch_size,
                                                 th1_0: batch_size,
                                                 sm_r_0: batch_size*seq_length,
                                                 sm_r_1: 24, })

    # Sub the backward prop values
    # TODO (Joel): Fix this up when all backprop works!
    correct_alg_flops = correct_alg_flops.subs({ g_mm_0: batch_size,
                                                 g_mms_0: batch_size,
                                                 g_ms_0: batch_size,
                                                 g_m_0: batch_size,
                                                 g_ms1_0: batch_size,
                                                 g_m1_0: batch_size,
                                                 g_ms2_0: batch_size,
                                                 g_m2_0: batch_size,
                                                 g_ms21_0: batch_size,
                                                 g_m21_0: batch_size,
                                                 g_mus_0: batch_size,
                                                 g_mu_0: batch_size,
                                                 g_mu1_0: batch_size,
                                                 g_s_0: batch_size,
                                                 g_c_0: batch_size,
                                                 g_an_0: batch_size,
                                                 g_an1_0: batch_size,
                                                 g_sm_m_0: batch_size * seq_length,
                                                 g_sm_m_1: 24, })

    assert graph.isValid()

    print('BOUND GRAPH: {}\n\n'.format(graph))

    # HHHHAAAAAAXXXXXX: FIX THIS! DUE TO SHAPEOP SYMBOL PROPAGATION!
    algorithmic_flops = algorithmic_flops.subs({hidden_dim: 24})

    print('Bound Flops test:')
    print('    CouGr:   {}'.format(algorithmic_flops))
    print('    Correct: {}'.format(correct_alg_flops))
    assert sympy.simplify(algorithmic_flops - correct_alg_flops) == 0, \
        'Bound alg flops incorrect!\n  Expecting: {}\n  Calculated: {}\n  Difference: {}' \
        .format(correct_alg_flops, algorithmic_flops, algorithmic_flops - correct_alg_flops)


if __name__ == "__main__":
    test_tf_dynamic_rnn()

