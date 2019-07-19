import re
import sympy
import catamount.frameworks.tensorflow
from catamount.api import utils
from catamount.tensors.tensor_shape import TensorShape


tf_example_filename = 'catamount/frameworks/example_graphs/tensorflow/rnn/output_dynamic_rnn_with_backprop/tf_graph.meta'

def test_tf_dynamic_rnn():
    graph = catamount.frameworks.tensorflow.import_graph(tf_example_filename)

    print('INITIAL GRAPH: {}\n\n'.format(graph))

    assert graph.isValid()

    algorithmic_flops = graph.calcAlgFlops()

    # Symbols we will use
    batch_size = utils.getIntSymbolFromString('batch_size')
    seq_length = utils.getIntSymbolFromString('seq_length')
    hidden_dim = utils.getIntSymbolFromString('hidden_dim')

    graph_iters = utils.getIntSymbolFromString('graph::iters')
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
    forward_flops = rwb_iters * (hidden_dim * a_0 + hidden_dim * a1_0 + 4 * hidden_dim * ba_0 + 16 * hidden_dim**2 * mm_0 + \
                                 hidden_dim * m_0 + hidden_dim * m1_0 + hidden_dim * m2_0 + 4 * hidden_dim * s_0 + \
                                 4 * hidden_dim * s1_0 + 4 * hidden_dim * s2_0 + 6 * hidden_dim * th_0 + 6 * hidden_dim * th1_0 + 6) + \
                    3 * sm_r_0 * sm_r_1 + \
                    16 * hidden_dim**2 + 8 * hidden_dim + 3
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
    backward_flops = grad_iters * (hidden_dim * g_an_0 + 3 * hidden_dim * g_an1_0 + 16 * hidden_dim**2 * g_mm_0 + 16 * hidden_dim**2 * g_mms_0 + \
                                   3 * hidden_dim * g_ms_0 + 2 * hidden_dim * g_m_0 + 3 * hidden_dim * g_ms1_0 + 2 * hidden_dim * g_m1_0 + \
                                   3 * hidden_dim * g_ms2_0 + 2 * hidden_dim * g_m2_0 + 3 * hidden_dim * g_ms21_0 + 2 * hidden_dim * g_m21_0 + \
                                   3 * hidden_dim * g_mus_0 + 2 * hidden_dim * g_mu_0 + 2 * hidden_dim * g_mu1_0 + 2 * hidden_dim * g_s_0 + \
                                   4 * hidden_dim * g_c_0 + 8 * hidden_dim**2 + 4 * hidden_dim + 2) + \
                     g_sm_m_0 * g_sm_m_1
    general_correct_alg_flops = forward_flops + backward_flops
    correct_alg_flops = general_correct_alg_flops.subs({hidden_dim: 24})

    # Now, bind tensor names in the graph and verify that the algorithmic
    # Flop counts reflect the new name bindings
    print('Loaded Flops test:')
    print('    Catamount:   {}'.format(algorithmic_flops))
    print('    Correct: {}'.format(correct_alg_flops))
    assert sympy.simplify(algorithmic_flops - correct_alg_flops) == 0, \
        'Initial alg flops incorrect!\n  Expecting: {}\n  Calculated: {}' \
        .format(correct_alg_flops, algorithmic_flops)

    # Manually set some variables
    # TODO (Joel): Fix this up when all tensor arrays work!
    ta_op = graph.opsByName['rnn/TensorArrayStack/TensorArraySizeV3']
    ta_op._outputs[0].setValue(seq_length)
    ta_op = graph.opsByName['rnn/TensorArrayStack/TensorArrayGatherV3']
    ta_op._outputs[0].mergeShape([seq_length, batch_size, hidden_dim], make_symbolic=True)
    ta_op = graph.opsByName['rnn/while/TensorArrayReadV3']
    ta_op._outputs[0].mergeShape([batch_size, hidden_dim], make_symbolic=True)

    ta_op = graph.opsByName['Gradient/Compute/gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3']
    ta_op._outputs[0].mergeShape([batch_size, hidden_dim], make_symbolic=True)

    # Bind constant values first
    const_dict = { # Store the shapes of certain tensors as constants
                   'rnn/Const': [hidden_dim],
                   'rnn/Const_1': [hidden_dim],
                 }
    graph.bindConstantValues(const_dict)

    # NOTE: This also works: batch_size = 'batch_size'
    # Bind placeholders (a and b) output dimensions 0 to name batch_size
    bind_dict = { # Variables
                  'rnn/basic_lstm_cell/kernel': [2 * hidden_dim, 4 * hidden_dim],
                  'rnn/basic_lstm_cell/bias': [4 * hidden_dim],
                  # Placeholders
                  'a': [batch_size, seq_length, hidden_dim],
                  'c_init_state': [batch_size, hidden_dim],
                  'h_init_state': [batch_size, hidden_dim],
                  'out_correct': [batch_size, seq_length],
                  # Constants
                  'Gradient/Compute/gradients/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc': [2 * hidden_dim, 4 * hidden_dim],
                  'Gradient/Compute/gradients/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc': [4 * hidden_dim],
                }
    graph.bindShapesAndPropagate(bind_dict, make_symbolic=True, warn_if_ill_defined=True)

    algorithmic_flops = graph.calcAlgFlops()

    # Update the algorithmic Flops formula
    # Sub the forward prop values
    correct_alg_flops = general_correct_alg_flops.subs({ ba_0: batch_size,
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
                                                 sm_r_1: hidden_dim, })

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
                                                 g_sm_m_1: hidden_dim, })

    assert graph.isValid()

    print('BOUND GRAPH:\n{}\n\n'.format(graph))

    print('Bound Flops test:')
    print('    Catamount:   {}'.format(algorithmic_flops))
    print('    Correct: {}'.format(correct_alg_flops))
    assert sympy.simplify(algorithmic_flops - correct_alg_flops) == 0, \
        'Bound alg flops incorrect!\n  Expecting: {}\n  Calculated: {}\n  Difference: {}' \
        .format(correct_alg_flops, algorithmic_flops, algorithmic_flops - correct_alg_flops)

    bind_subs = { # Symbols/names to bind in next tests:
                  graph_iters: 1,
                  rwb_iters: seq_length,
                  grad_iters: seq_length,
                  batch_size: 128,
                  seq_length: 32,
                  hidden_dim: 256,
                }

    print('\n\nBound values')
    print('Symbol associations: {}\n'.format(bind_subs))

    # Calculate model parameter count
    parameters = graph.calcModelParameters()
    resolved_params = parameters.subs(bind_subs)
    try:
        resolved_params = int(resolved_params)
    except:
        print('ERROR: resolved_params should be int, but is {} = {}'.format(
              type(resolved_params), resolved_params))
    correct_params = 525312
    assert resolved_params == correct_params, \
           'Incorrect model params: {}'.format(resolved_params)
    print('Parameters: {}\nWith specified dims: {}\n'.format(parameters, resolved_params))

    # Calculate algorithmic Flops
    alg_flops = graph.calcAlgFlops()
    resolved_flops = alg_flops.subs(bind_subs)
    try:
        resolved_flops = int(resolved_flops)
    except:
        print('ERROR: resolved_flops should be int, but is {} = {}'.format(
              type(resolved_flops), resolved_flops))
    correct_flops = 12980357379
    assert resolved_flops == correct_flops, \
           'Incorrect algorithmic flops: {}'.format(resolved_flops)
    print('Algorithmic Flops: {}\nWith specified dims: {}\n'.format(alg_flops, resolved_flops))

    # Calculate algorthmic Bytes accessed
    alg_bytes = graph.calcAlgBytes()
    resolved_bytes = alg_bytes.subs(bind_subs)
    try:
        resolved_bytes = int(resolved_bytes)
    except:
        print('ERROR: resolved_bytes should be int, but is {} = {}'.format(
              type(resolved_bytes), resolved_bytes))
    correct_bytes = 1134017252
    assert resolved_bytes == correct_bytes, \
           'Incorrect algorithmic bytes: {}'.format(resolved_bytes)
    print('Alg bytes accessed: {}\nWith specified dims: {}\n'.format(alg_bytes, resolved_bytes))

    # Calculate algorthmic Bytes accessed
    alg_footprint = graph.calcAlgFootprint()
    resolved_footprint = alg_footprint.subs(bind_subs)
    try:
        resolved_footprint = int(resolved_footprint)
    except:
        print('ERROR: resolved_footprint should be int, but is {} = {}'.format(
              type(resolved_footprint), resolved_footprint))
    correct_footprint = 441447784
    assert resolved_footprint == correct_footprint, \
           'Incorrect algorithmic footprint: {}'.format(resolved_footprint)
    print('Alg mem footprint: {}\nWith specified dims: {}\n'.format(alg_footprint, resolved_footprint))

    # Calculate the minimal memory footprint for a step
    alg_min_footprint = graph.calcMinimalFootprint(symbol_subs=bind_subs)
    resolved_min_footprint = alg_min_footprint
    try:
        resolved_min_footprint = int(resolved_min_footprint)
    except:
        print('ERROR: resolved_min_footprint should be int, but is {} = {}'.format(
              type(resolved_min_footprint), resolved_min_footprint))
    correct_min_footprint = 38153640
    error_percent = abs(correct_min_footprint - resolved_min_footprint) / correct_min_footprint
    if error_percent > 0.15:
        print('Incorrect algorithmic footprint: {} (err: {})!'.format(resolved_min_footprint, error_percent))
    print('Alg minimal footprint: {}\nWith specified dims: {} (err: {})\n'.format(alg_footprint, resolved_min_footprint, error_percent))

    # Calculate algorithmic IO per step
    total_io_footprint = 0
    for op in graph.getPlaceholders():
        total_io_footprint += op.calcAlgFootprint()
    resolved_io_footprint = total_io_footprint.subs(bind_subs)
    print('Alg IO footprint: {}\nWith specified dims: {}\n'.format(total_io_footprint, resolved_io_footprint))


if __name__ == "__main__":
    test_tf_dynamic_rnn()

