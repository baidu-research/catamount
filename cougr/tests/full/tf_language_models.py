import argparse
import numpy as np
import pickle
import sympy
import sys
sys.setrecursionlimit(50000)

from cougr.api import utils
import cougr.frameworks.tensorflow
from cougr.ops.constant import *
from cougr.ops.unknown_op import UnknownOp
from cougr.ops.variable import *


is_pytest_run = False

def test_tf_word_language_model():
    global is_pytest_run
    is_pytest_run = True

    run_tf_language_model(domain='wordlm')

def test_tf_character_language_model():
    global is_pytest_run
    is_pytest_run = True

    run_tf_language_model(domain='charlm')

def test_tf_machine_translation_model():
    global is_pytest_run
    is_pytest_run = True

    run_tf_language_model(domain='nmt')

def run_tf_language_model(domain=None, build_projection=False):
    global is_pytest_run

    if domain == 'wordlm':
        graph_meta = 'cougr/frameworks/example_graphs/tensorflow/full_models/language_models/word_lm_n2004_l2_sgd_lr0.2_nodrop_b128_v10k_d20_s80-best_model.meta'
    elif domain == 'charlm':
        graph_meta = 'cougr/frameworks/example_graphs/tensorflow/full_models/language_models/char_lm_n2004_l10_sgd_lr0.15_rhn_b128_vchar_d1.0_s150-latest_model.meta'
    elif domain == 'nmt':
        graph_meta = 'cougr/frameworks/example_graphs/tensorflow/full_models/language_models/nmt_el2_dl1_n1024_b128-translate.ckpt-1000.meta'
    else:
        raise NotImplementedError('ERROR: Unknown domain: {}'.format(domain))

    graph = cougr.frameworks.tensorflow.import_graph(graph_meta)
    assert graph.isValid()

    # ============ TO REMOVE INITIALIZATION OPS! =============
    # NOTE: This code is pretty general and is likely to be migrated into
    # CouGr code for removing TF-specific initialization ops
    from cougr.ops import AssignOp
    from cougr.ops import VariableOp
    assign_ops = set()
    for op in graph.opsByName.values():
        if isinstance(op, AssignOp):
            assign_ops.add(op)
    for assign_op in assign_ops:
        my_ancestors = set()
        my_frontier = set()
        my_frontier.add(assign_op)
        while len(my_frontier) > 0:
            next_op = my_frontier.pop()
            for in_tensor in next_op.inputs:
                if not isinstance(in_tensor.producer, VariableOp):
                    my_frontier.add(in_tensor.producer)
            my_ancestors.add(next_op)
        for next_op in my_ancestors:
            graph.removeOp(next_op)
    assert graph.isValid()

    # Next, remove ops that are not executed during a standard training step:
    if domain == 'wordlm':
        graph_ops = list(graph._ops_by_name.values())
        for op in graph_ops:
             # Certain ops are only used for inference
             if 'Model/Recurrent_1_lstm_3/' in op.name or \
                'Model/Recurrent_2_lstm_3/' in op.name or \
                'Model/FullSoftmaxLoss_1_3/' in op.name or \
                'Model/Collapse_1/' in op.name or \
                'Model/Embedding_1_3/' in op.name or \
                'Model/Labels_1/' in op.name or \
                'Model/Mask_1/' in op.name:
                 graph.removeOp(op)
             elif op.name == 'Model/Sum_1' or \
                  op.name == 'Model/Const_1' or \
                  op.name == 'Model/truediv_1' or \
                  op.name == 'Model/MPIAllreduce_2' or \
                  op.name == 'Model/MPIAllreduce_3' or \
                  op.name == 'Model/div_2' or \
                  op.name == 'Model/div_3' or \
                  op.name == 'Model/MPISize_2' or \
                  op.name == 'Model/MPISize_3' or \
                  op.name == 'Model/Cast_3' or \
                  op.name == 'Model/Cast_4' or \
                  op.name == 'Model/Cast_5' or \
                  op.name == 'Model/Exp_1':
                 graph.removeOp(op)
    elif domain == 'charlm':
        graph_ops = list(graph._ops_by_name.values())
        for op in graph_ops:
             # Certain ops are only used for inference
             if 'Model/Recurrent_1_rhn_3/' in op.name or \
                'Model/FullSoftmaxLoss_1_3/' in op.name or \
                'Model/Collapse_1/' in op.name or \
                'Model/Embedding_1_3/' in op.name or \
                'Model/Labels_1/' in op.name or \
                'Model/Mask_1/' in op.name:
                 graph.removeOp(op)
             elif op.name == 'Model/Sum_1' or \
                  op.name == 'Model/Const_1' or \
                  op.name == 'Model/Size_1' or \
                  op.name == 'Model/Cast_5' or \
                  op.name == 'Model/Cast_8' or \
                  op.name == 'Model/Cast_9' or \
                  op.name == 'Model/truediv_2' or \
                  op.name == 'Model/truediv_3' or \
                  op.name == 'Model/MPIAllreduce_4' or \
                  op.name == 'Model/MPIAllreduce_5' or \
                  op.name == 'Model/MPIAllreduce_6' or \
                  op.name == 'Model/MPIAllreduce_7' or \
                  op.name == 'Model/div_4' or \
                  op.name == 'Model/div_5' or \
                  op.name == 'Model/div_6' or \
                  op.name == 'Model/div_7' or \
                  op.name == 'Model/MPISize_6' or \
                  op.name == 'Model/MPISize_7' or \
                  op.name == 'Model/Exp_1':
                 graph.removeOp(op)
    elif domain == 'nmt':
        pass
    else:
        raise NotImplementedError('ERROR: Unknown domain: {}'.format(domain))

    if not is_pytest_run:
        print('Initial graph:\n{}\n'.format(graph))
    init_params = graph.calcModelParameters()
    print('Initial parameters: {}'.format(init_params))
    print('Initial Flops: {}\n'.format(graph.calcAlgFlops()))

    print('Placeholders:')
    for op in graph.getPlaceholders():
        print(op.debugString())
    print('')

    # Set up symbols to name dimensions
    hidden_dim_symbol = utils.getIntSymbolFromString('hidden_dim')
    vocab_size_symbol = utils.getIntSymbolFromString('vocab_size')
    subbatch_size_symbol = utils.getIntSymbolFromString('subbatch_size')
    sequence_length_symbol = utils.getIntSymbolFromString('sequence_length')
    batch_times_seq_symbol = sequence_length_symbol * subbatch_size_symbol
    graph_iters_symbol = utils.getIntSymbolFromString('graph::iters')

    # Convert these constant dimensions to symbols
    base_subbatch_size = None
    base_sequence_length = None
    if domain == 'wordlm':
        base_hidden_dim = 2004
        base_vocab_size = 10004
    elif domain == 'charlm':
        base_hidden_dim = 2004
        base_vocab_size = 98
    elif domain == 'nmt':
        base_hidden_dim = 1024
        base_vocab_size = 36548
        base_sequence_length = 19
        base_subbatch_size = 128
    else:
        raise NotImplementedError('ERROR: Unknown domain: {}'.format(domain))

    # HAXXX: Manually setting TensorArray and StackPop shapes!
    if domain == 'wordlm' or domain == 'charlm' or domain == 'nmt':
        for op in graph._ops_by_name.values():
            op_name_suffix = op.name.split('/')[-1]
            if 'StackPop' in op_name_suffix:
                # HAXXXX: Just verify op structure. Pull shapes and values
                # from corresponding StackPush ops below
                assert isinstance(op, UnknownOp)
                assert len(op._inputs) == 1
                assert len(op._outputs) == 1
                continue
            elif 'TensorArrayGather' in op_name_suffix:
                assert isinstance(op, UnknownOp)
                assert len(op._inputs) == 3
                assert len(op._outputs) == 1
                if domain == 'wordlm' or domain == 'charlm':
                    assert op._outputs[0].shape.isUnknown() or \
                           op._outputs[0].shape.rank == 3, \
                           '{}'.format(op.name)
                    gather_shape = [sequence_length_symbol,
                                    subbatch_size_symbol,
                                    hidden_dim_symbol]
                else:
                    assert domain == 'nmt'
                    assert op._outputs[0].shape.isUnknown() or \
                           op._outputs[0].shape.rank == 2 or \
                           op._outputs[0].shape.rank == 3, \
                           '{}'.format(op.name)
                    if not op._outputs[0].shape.isUnknown():
                        if op._outputs[0].shape.rank == 3:
                            out_shape = [base_sequence_length,
                                         base_subbatch_size,
                                         base_hidden_dim]
                            # Verify that the shape is clearly specified
                            op._outputs[0].shape.mergeShape(out_shape, make_symbolic=True)
                            gather_shape = [sequence_length_symbol,
                                            subbatch_size_symbol,
                                            hidden_dim_symbol]
                        else:
                            # This TAGather is known to be unused, so who cares?!
                            assert len(op._outputs[0].consumers) == 0
                            continue
                op._outputs[0].shape.mergeShape(gather_shape, make_symbolic=True)
                for idx in range(op._outputs[0].shape.rank):
                    op._outputs[0].shape.dims[idx]._value = None
            elif 'TensorArraySize' in op_name_suffix:
                assert isinstance(op, UnknownOp)
                assert len(op._inputs) == 2
                assert len(op._outputs) == 1
                assert op._outputs[0].shape.rank == 0
                op._outputs[0].setValue(sequence_length_symbol)
            elif 'TensorArrayRead' in op_name_suffix:
                assert isinstance(op, UnknownOp)
                assert len(op._inputs) == 3
                assert len(op._outputs) == 1
                assert op._outputs[0].shape.isUnknown() or \
                       op._outputs[0].shape.rank == 2, \
                       '{}'.format(op.name)
                if not op._outputs[0].shape.isUnknown():
                    assert op._outputs[0].shape.dims[1].value == base_hidden_dim
                read_shape = [subbatch_size_symbol, hidden_dim_symbol]
                op._outputs[0].shape.mergeShape(read_shape, make_symbolic=True)
                for idx in range(op._outputs[0].shape.rank):
                    op._outputs[0].shape.dims[idx]._value = None
    else:
        raise NotImplementedError('ERROR: Unknown domain: {}'.format(domain))

    assert graph.isValid()

    if domain == 'wordlm':
        const_dict = {
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_lstm_1/rnn/while/rnn/BiasAdd/Enter_grad/b_acc': [4 * hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_lstm_1/rnn/while/rnn/MatMul/Enter_grad/b_acc': [2 * hidden_dim_symbol, 4 * hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_2_lstm_1/rnn/while/rnn/BiasAdd/Enter_grad/b_acc': [4 * hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_2_lstm_1/rnn/while/rnn/MatMul/Enter_grad/b_acc': [2 * hidden_dim_symbol, 4 * hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_lstm_1/rnn/while/rnn/BiasAdd/Enter_grad/b_acc': [4 * hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_lstm_1/rnn/while/rnn/MatMul/Enter_grad/b_acc': [2 * hidden_dim_symbol, 4 * hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_2_lstm_1/rnn/while/rnn/BiasAdd/Enter_grad/b_acc': [4 * hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_2_lstm_1/rnn/while/rnn/MatMul/Enter_grad/b_acc': [2 * hidden_dim_symbol, 4 * hidden_dim_symbol],
                    }
    elif domain == 'charlm':
        const_dict = {
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/h_0/h_0/BiasAdd/Enter_grad/b_acc': [hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/h_0/h_0/MatMul/Enter_grad/b_acc': [2 * hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/h_1/h_1/BiasAdd/Enter_grad/b_acc': [hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/h_1/h_1/MatMul/Enter_grad/b_acc': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/h_2/h_2/BiasAdd/Enter_grad/b_acc': [hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/h_2/h_2/MatMul/Enter_grad/b_acc': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/h_3/h_3/BiasAdd/Enter_grad/b_acc': [hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/h_3/h_3/MatMul/Enter_grad/b_acc': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/h_4/h_4/BiasAdd/Enter_grad/b_acc': [hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/h_4/h_4/MatMul/Enter_grad/b_acc': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/h_5/h_5/BiasAdd/Enter_grad/b_acc': [hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/h_5/h_5/MatMul/Enter_grad/b_acc': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/h_6/h_6/BiasAdd/Enter_grad/b_acc': [hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/h_6/h_6/MatMul/Enter_grad/b_acc': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/h_7/h_7/BiasAdd/Enter_grad/b_acc': [hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/h_7/h_7/MatMul/Enter_grad/b_acc': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/h_8/h_8/BiasAdd/Enter_grad/b_acc': [hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/h_8/h_8/MatMul/Enter_grad/b_acc': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/h_9/h_9/BiasAdd/Enter_grad/b_acc': [hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/h_9/h_9/MatMul/Enter_grad/b_acc': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/t_0/t_0/BiasAdd/Enter_grad/b_acc': [hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/t_0/t_0/MatMul/Enter_grad/b_acc': [2 * hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/t_1/t_1/BiasAdd/Enter_grad/b_acc': [hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/t_1/t_1/MatMul/Enter_grad/b_acc': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/t_2/t_2/BiasAdd/Enter_grad/b_acc': [hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/t_2/t_2/MatMul/Enter_grad/b_acc': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/t_3/t_3/BiasAdd/Enter_grad/b_acc': [hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/t_3/t_3/MatMul/Enter_grad/b_acc': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/t_4/t_4/BiasAdd/Enter_grad/b_acc': [hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/t_4/t_4/MatMul/Enter_grad/b_acc': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/t_5/t_5/BiasAdd/Enter_grad/b_acc': [hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/t_5/t_5/MatMul/Enter_grad/b_acc': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/t_6/t_6/BiasAdd/Enter_grad/b_acc': [hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/t_6/t_6/MatMul/Enter_grad/b_acc': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/t_7/t_7/BiasAdd/Enter_grad/b_acc': [hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/t_7/t_7/MatMul/Enter_grad/b_acc': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/t_8/t_8/BiasAdd/Enter_grad/b_acc': [hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/t_8/t_8/MatMul/Enter_grad/b_acc': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/t_9/t_9/BiasAdd/Enter_grad/b_acc': [hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Recurrent_1_rhn_1/rnn/while/t_9/t_9/MatMul/Enter_grad/b_acc': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/h_0/Bias/Initializer/Const': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/h_1/Bias/Initializer/Const': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/h_2/Bias/Initializer/Const': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/h_3/Bias/Initializer/Const': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/h_4/Bias/Initializer/Const': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/h_5/Bias/Initializer/Const': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/h_6/Bias/Initializer/Const': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/h_7/Bias/Initializer/Const': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/h_8/Bias/Initializer/Const': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/h_9/Bias/Initializer/Const': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/t_0/Bias/Initializer/Const': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/t_1/Bias/Initializer/Const': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/t_2/Bias/Initializer/Const': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/t_3/Bias/Initializer/Const': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/t_4/Bias/Initializer/Const': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/t_5/Bias/Initializer/Const': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/t_6/Bias/Initializer/Const': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/t_7/Bias/Initializer/Const': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/t_8/Bias/Initializer/Const': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/t_9/Bias/Initializer/Const': [hidden_dim_symbol],
                    }
    elif domain == 'nmt':
        const_dict = {
                      'gradients/dynamic_seq2seq/decoder/decoder/while/BasicDecoderStep/decoder/attention/attention_layer/MatMul/Enter_grad/b_acc': [3 * hidden_dim_symbol, hidden_dim_symbol],
                      'gradients/dynamic_seq2seq/decoder/decoder/while/BasicDecoderStep/decoder/attention/basic_lstm_cell/BiasAdd/Enter_grad/b_acc': [4 * hidden_dim_symbol],
                      'gradients/dynamic_seq2seq/decoder/decoder/while/BasicDecoderStep/decoder/attention/basic_lstm_cell/MatMul/Enter_grad/b_acc': [3 * hidden_dim_symbol, 4 * hidden_dim_symbol],
                      'gradients/dynamic_seq2seq/encoder/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc': [4 * hidden_dim_symbol],
                      'gradients/dynamic_seq2seq/encoder/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc': [2 * hidden_dim_symbol, 4 * hidden_dim_symbol],
                      'gradients/dynamic_seq2seq/encoder/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc': [4 * hidden_dim_symbol],
                      'gradients/dynamic_seq2seq/encoder/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc': [2 * hidden_dim_symbol, 4 * hidden_dim_symbol],
                    }
    else:
        raise NotImplementedError('Manually set constant op shapes for domain {}'.format(domain))

    for const_key, const_shape in const_dict.items():
        try:
            if const_key in graph.opsByName.keys():
                const_op = graph.opsByName[const_key]
                assert isinstance(const_op, ConstantOp)
                const_op._outputs[0].shape.mergeShape(const_shape, make_symbolic=True)
                if const_op._outputs[0].value is not None:
                    const_op._outputs[0]._value = None
            else:
                print('WARN: ConstantOp not found: {}'.format(const_key))
        except Exception as exc:
            print('WARN: ConstantOp unknown problem: {}: {}'.format(const_key, exc))

    if domain == 'wordlm':
        const_dict = {
                      'Model/Collapse/Reshape/shape': [-1, hidden_dim_symbol],
                      'Model/Recurrent_1_lstm_1/rnn/Const': [hidden_dim_symbol],
                      'Model/Recurrent_1_lstm_1/rnn/Const_1': [hidden_dim_symbol],
                      'Model/Recurrent_2_lstm_1/rnn/Const': [hidden_dim_symbol],
                      'Model/Recurrent_2_lstm_1/rnn/Const_1': [hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/Embedding_1_1/Gather_grad/Shape': [vocab_size_symbol, hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/FullSoftmaxLoss_1_1/add_grad/Shape_1': [1, vocab_size_symbol],
                    }
    elif domain == 'charlm':
        const_dict = {
                      'Model/Collapse/Reshape/shape': [-1, hidden_dim_symbol],
                      'Model/Recurrent_1_rhn_1/rnn/Const': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn_1/rnn/Const_1': [hidden_dim_symbol],
                      'Model/Gradient/Compute/gradients/Model/FullSoftmaxLoss_1_1/add_grad/Shape_1': [1, vocab_size_symbol],
                      'Model/Gradient/Compute/gradients/Model/Embedding_1_1/Gather_grad/Shape': [vocab_size_symbol, hidden_dim_symbol],
                    }
    elif domain == 'nmt':
        const_dict = {
                       'gradients/dynamic_seq2seq/decoder/output_projection/Tensordot/Reshape_1_grad/Shape': [hidden_dim_symbol, vocab_size_symbol],
                       'gradients/dynamic_seq2seq/decoder/embedding_lookup_grad/Shape': [vocab_size_symbol, hidden_dim_symbol],
                       'gradients/dynamic_seq2seq/encoder/embedding_lookup_grad/Shape': [vocab_size_symbol, hidden_dim_symbol],
                       'gradients/dynamic_seq2seq/decoder/LuongAttention/memory_layer/Tensordot/Reshape_1_grad/Shape': [2 * hidden_dim_symbol, hidden_dim_symbol],
                       'dynamic_seq2seq/encoder/bidirectional_rnn/fw/fw/Const_1': [hidden_dim_symbol],
                       'dynamic_seq2seq/encoder/bidirectional_rnn/fw/fw/Const_4': [hidden_dim_symbol],
                       'dynamic_seq2seq/encoder/bidirectional_rnn/fw/fw/DeviceWrapperZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const': [hidden_dim_symbol],
                       'dynamic_seq2seq/encoder/bidirectional_rnn/fw/fw/DeviceWrapperZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1': [hidden_dim_symbol],
                       'dynamic_seq2seq/encoder/bidirectional_rnn/fw/fw/DeviceWrapperZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_2': [hidden_dim_symbol],
                       'dynamic_seq2seq/encoder/bidirectional_rnn/fw/fw/DeviceWrapperZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_3': [hidden_dim_symbol],
                       'dynamic_seq2seq/encoder/bidirectional_rnn/bw/bw/Const_1': [hidden_dim_symbol],
                       'dynamic_seq2seq/encoder/bidirectional_rnn/bw/bw/Const_4': [hidden_dim_symbol],
                       'dynamic_seq2seq/encoder/bidirectional_rnn/bw/bw/DeviceWrapperZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const': [hidden_dim_symbol],
                       'dynamic_seq2seq/encoder/bidirectional_rnn/bw/bw/DeviceWrapperZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1': [hidden_dim_symbol],
                       'dynamic_seq2seq/encoder/bidirectional_rnn/bw/bw/DeviceWrapperZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_2': [hidden_dim_symbol],
                       'dynamic_seq2seq/encoder/bidirectional_rnn/bw/bw/DeviceWrapperZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_3': [hidden_dim_symbol],
                       'dynamic_seq2seq/decoder/output_projection/Tensordot/Reshape_1/shape': [hidden_dim_symbol, vocab_size_symbol],
                       'dynamic_seq2seq/decoder/output_projection/Tensordot/Const_2': [vocab_size_symbol],
                       'dynamic_seq2seq/decoder/decoder/Const': [hidden_dim_symbol],
                       'dynamic_seq2seq/decoder/decoder/Const_1': [hidden_dim_symbol],
                       'dynamic_seq2seq/decoder/LuongAttention/memory_layer/Tensordot/Const_2': [hidden_dim_symbol],
                       'dynamic_seq2seq/decoder/LuongAttention/memory_layer/Tensordot/Reshape_1/shape': [2 * hidden_dim_symbol, hidden_dim_symbol],
                       'dynamic_seq2seq/decoder/DeviceWrapperZeroState/AttentionWrapperZeroState/Const': [hidden_dim_symbol],
                       'dynamic_seq2seq/decoder/DeviceWrapperZeroState/AttentionWrapperZeroState/Const_1': [hidden_dim_symbol],
                       'dynamic_seq2seq/decoder/DeviceWrapperZeroState/AttentionWrapperZeroState/DeviceWrapperZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const': [hidden_dim_symbol],
                       'dynamic_seq2seq/decoder/DeviceWrapperZeroState/AttentionWrapperZeroState/DeviceWrapperZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1': [hidden_dim_symbol],
                       'dynamic_seq2seq/decoder/DeviceWrapperZeroState/AttentionWrapperZeroState/DeviceWrapperZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_2': [hidden_dim_symbol],
                       'dynamic_seq2seq/decoder/DeviceWrapperZeroState/AttentionWrapperZeroState/DeviceWrapperZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_3': [hidden_dim_symbol],
                       'buffer_size': 256 * hidden_dim_symbol,
                       'buffer_size_1': 256 * hidden_dim_symbol,
                       'buffer_size_2': 125 * hidden_dim_symbol,
                       'buffer_size_3': 125 * hidden_dim_symbol,
                       'buffer_size_4': 125 * hidden_dim_symbol,
                       'buffer_size_5': 125 * hidden_dim_symbol,
                       'buffer_size_6': 125 * hidden_dim_symbol,
                       'buffer_size_7': 125 * hidden_dim_symbol,
                       'buffer_size_8': 125 * hidden_dim_symbol,

                     }
    else:
        raise NotImplementedError('Manually set constant op values for domain {}'.format(domain))

    for const_key, const_val in const_dict.items():
        try:
            if const_key in graph.opsByName.keys():
                const_op = graph.opsByName[const_key]
                assert isinstance(const_op, ConstantOp)
                const_op._outputs[0].setValue(const_val)
            else:
                print('WARN: ConstantOp not found: {}'.format(const_key))
        except Exception as exc:
            print('WARN: ConstantOp unknown problem: {}: {}'.format(const_key, exc))


    # Next, bind the placeholders and propagate shapes
    if domain == 'wordlm':
        bind_dict = { # Placeholders
                      'Input/Input': [subbatch_size_symbol, sequence_length_symbol],
                      'Labels/Labels': [subbatch_size_symbol, sequence_length_symbol],
                      'Model/Placeholder': [subbatch_size_symbol, hidden_dim_symbol],
                      'Model/Placeholder_1': [subbatch_size_symbol, hidden_dim_symbol],
                      'Model/Placeholder_2': [subbatch_size_symbol, hidden_dim_symbol],
                      'Model/Placeholder_3': [subbatch_size_symbol, hidden_dim_symbol],
                      'Model/Placeholder_4': [subbatch_size_symbol, hidden_dim_symbol],
                      'Model/Placeholder_5': [subbatch_size_symbol, hidden_dim_symbol],
                      'Model/Placeholder_6': [subbatch_size_symbol, hidden_dim_symbol],
                      'Model/Placeholder_7': [subbatch_size_symbol, hidden_dim_symbol],
                      # Variables
                      'Model/Embedding_1/EmbeddingWeights': [vocab_size_symbol, hidden_dim_symbol],
                      'Model/FullSoftmaxLoss_1/W_Softmax': [vocab_size_symbol, hidden_dim_symbol],
                      'Model/FullSoftmaxLoss_1/b_Softmax': [1, vocab_size_symbol],
                      'Model/Recurrent_1_lstm/rnn/Bias': [4 * hidden_dim_symbol],
                      'Model/Recurrent_1_lstm/rnn/Matrix': [2 * hidden_dim_symbol, 4 * hidden_dim_symbol],
                      'Model/Recurrent_2_lstm/rnn/Bias': [4 * hidden_dim_symbol],
                      'Model/Recurrent_2_lstm/rnn/Matrix': [2 * hidden_dim_symbol, 4 * hidden_dim_symbol],
                    }
    elif domain == 'charlm':
        bind_dict = { # Placeholders
                      'Input/Input': [subbatch_size_symbol, sequence_length_symbol],
                      'Labels/Labels': [subbatch_size_symbol, sequence_length_symbol],
                      'Model/Placeholder': [subbatch_size_symbol, hidden_dim_symbol],
                      'Model/Placeholder_1': [subbatch_size_symbol, hidden_dim_symbol],
                      # Variables
                      'Model/Embedding_1/EmbeddingWeights': [vocab_size_symbol, hidden_dim_symbol],
                      'Model/FullSoftmaxLoss_1/W_Softmax': [vocab_size_symbol, hidden_dim_symbol],
                      'Model/FullSoftmaxLoss_1/b_Softmax': [1, vocab_size_symbol],
                      'Model/Recurrent_1_rhn/rnn/h_0/Bias': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/h_0/Matrix': [2 * hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/h_1/Bias': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/h_1/Matrix': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/h_2/Bias': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/h_2/Matrix': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/h_3/Bias': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/h_3/Matrix': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/h_4/Bias': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/h_4/Matrix': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/h_5/Bias': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/h_5/Matrix': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/h_6/Bias': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/h_6/Matrix': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/h_7/Bias': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/h_7/Matrix': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/h_8/Bias': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/h_8/Matrix': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/h_9/Bias': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/h_9/Matrix': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/t_0/Bias': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/t_0/Matrix': [2 * hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/t_1/Bias': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/t_1/Matrix': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/t_2/Bias': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/t_2/Matrix': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/t_3/Bias': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/t_3/Matrix': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/t_4/Bias': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/t_4/Matrix': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/t_5/Bias': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/t_5/Matrix': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/t_6/Bias': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/t_6/Matrix': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/t_7/Bias': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/t_7/Matrix': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/t_8/Bias': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/t_8/Matrix': [hidden_dim_symbol, hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/t_9/Bias': [hidden_dim_symbol],
                      'Model/Recurrent_1_rhn/rnn/t_9/Matrix': [hidden_dim_symbol, hidden_dim_symbol],
                    }
    elif domain == 'nmt':
        # HAX: Manually hack the iterator op
        it_op = graph.opsByName['IteratorGetNext']
        it_op._outputs[0].shape.mergeShape([subbatch_size_symbol, sequence_length_symbol], make_symbolic=True)
        it_op._outputs[1].shape.mergeShape([subbatch_size_symbol, sequence_length_symbol], make_symbolic=True)
        it_op._outputs[2].shape.mergeShape([subbatch_size_symbol, sequence_length_symbol], make_symbolic=True)
        it_op._outputs[3].shape.mergeShape([subbatch_size_symbol], make_symbolic=True)
        it_op._outputs[4].shape.mergeShape([subbatch_size_symbol], make_symbolic=True)

        bind_dict = { # Placeholders
                      # Variables
                      'dynamic_seq2seq/decoder/attention/attention_layer/kernel': [3 * hidden_dim_symbol, hidden_dim_symbol],
                      'dynamic_seq2seq/decoder/attention/basic_lstm_cell/bias': [4 * hidden_dim_symbol],
                      'dynamic_seq2seq/decoder/attention/basic_lstm_cell/kernel': [3 * hidden_dim_symbol, 4 * hidden_dim_symbol],
                      'dynamic_seq2seq/decoder/memory_layer/kernel': [2 * hidden_dim_symbol, hidden_dim_symbol],
                      'dynamic_seq2seq/decoder/output_projection/kernel': [hidden_dim_symbol, vocab_size_symbol],
                      'dynamic_seq2seq/encoder/bidirectional_rnn/bw/basic_lstm_cell/bias': [4 * hidden_dim_symbol],
                      'dynamic_seq2seq/encoder/bidirectional_rnn/bw/basic_lstm_cell/kernel': [2 * hidden_dim_symbol, 4 * hidden_dim_symbol],
                      'dynamic_seq2seq/encoder/bidirectional_rnn/fw/basic_lstm_cell/bias': [4 * hidden_dim_symbol],
                      'dynamic_seq2seq/encoder/bidirectional_rnn/fw/basic_lstm_cell/kernel': [2 * hidden_dim_symbol, 4 * hidden_dim_symbol],
                      'embeddings/decoder/embedding_decoder': [vocab_size_symbol, hidden_dim_symbol],
                      'embeddings/encoder/embedding_encoder': [vocab_size_symbol, hidden_dim_symbol],
                    }
    else:
        raise NotImplementedError('ERROR: Unknown domain: {}'.format(domain))

    print('Binding variables')

    graph.bindTensorShapeDimensions(bind_dict, warn_if_ill_defined=(not is_pytest_run), make_symbolic=True)
    assert graph.isValid()

    # Finally, more hacking... StackPops can pull from their corresponding
    # StackPushs. Try to propagate their shapes and/or values if possible
    # NOTE: This code is pretty general and is likely to be migrated into
    # CouGr ops for stacks later
    for op in graph._ops_by_name.values():
        op_name_split = op.name.split('/')
        if 'StackPop' in op_name_split[-1]:
            push_name_split = list(op_name_split)
            push_name_split[-1] = push_name_split[-1].replace('StackPop',
                                                              'StackPush')
            push_name = '/'.join(push_name_split)
            if push_name not in graph._ops_by_name.keys():
                print('WARN: CANNOT FIND CORRESPONDING STACK PUSH: {}'
                      .format(push_name))
                continue
            push_op = graph._ops_by_name[push_name]
            # Verify StackPush input[1].shape == StackPop output[0].shape
            assert push_op._inputs[1].shape == op._outputs[0].shape
            op._outputs[0].shape.mergeShape(push_op._inputs[1].shape, make_symbolic=True)
            if push_op._inputs[1].value is not None:
                op._outputs[0].setValue(push_op._inputs[1].value)

    graph.bindTensorShapeDimensions(bind_dict, warn_if_ill_defined=(not is_pytest_run), make_symbolic=True)
    assert graph.isValid()

    num_workers_symbol = utils.getIntSymbolFromString('num_workers')
    num_sampled_vocab_symbol = subbatch_size_symbol * sequence_length_symbol
    if domain == 'wordlm':
        base_sequence_length = 80
        base_subbatch_size = 64
        base_num_sampled_vocab = base_subbatch_size * base_sequence_length
        bind_str_subs = {
            'Model/Collapse_1/boolean_mask/Reshape_1:0::num_true': sequence_length_symbol * subbatch_size_symbol,
            'Model/Collapse/boolean_mask/Reshape_1:0::num_true': sequence_length_symbol * subbatch_size_symbol,
            'Model/Labels_1/boolean_mask/Reshape_1:0::num_true': sequence_length_symbol * subbatch_size_symbol,
            'Model/Labels/boolean_mask/Reshape_1:0::num_true': sequence_length_symbol * subbatch_size_symbol,
            'Model/Gradient/Compute/allgather_1::num_workers': num_workers_symbol,
            'Model/Gradient/Compute/allgather::num_workers': num_workers_symbol,
            'Model/Gradient/Compute/allgather_sizing_1::num_workers': num_workers_symbol,
            'Model/Gradient/Compute/allgather_sizing::num_workers': num_workers_symbol,
            'Model/Gradient/Compute/gradients/b_count_2_block::iters': sequence_length_symbol,
            'Model/Gradient/Compute/gradients/b_count_6_block::iters': sequence_length_symbol,
            'Model/Recurrent_1_lstm_1/rnn/while/LoopCond_block::iters': sequence_length_symbol,
            'Model/Recurrent_1_lstm_3/rnn/while/LoopCond_block::iters': sequence_length_symbol,
            'Model/Recurrent_2_lstm_1/rnn/while/LoopCond_block::iters': sequence_length_symbol,
            'Model/Recurrent_2_lstm_3/rnn/while/LoopCond_block::iters': sequence_length_symbol,
        }
    elif domain == 'charlm':
        base_sequence_length = 150
        base_subbatch_size = 128
        bind_str_subs = {
            'Model/Collapse/boolean_mask/Reshape_1:0::num_true': sequence_length_symbol * subbatch_size_symbol,
            'Model/Labels/boolean_mask/Reshape_1:0::num_true': sequence_length_symbol * subbatch_size_symbol,
            'Model/Recurrent_1_rhn_1/rnn/while/LoopCond_block::iters': sequence_length_symbol,
            'Model/Gradient/Compute/gradients/b_count_2_block::iters': sequence_length_symbol,
        }
    elif domain == 'nmt':
        bind_str_subs = {
            'dynamic_seq2seq/decoder/decoder/while/LoopCond_block::iters': sequence_length_symbol,
            'dynamic_seq2seq/encoder/bidirectional_rnn/bw/bw/while/LoopCond_block::iters': sequence_length_symbol,
            'dynamic_seq2seq/encoder/bidirectional_rnn/fw/fw/while/LoopCond_block::iters': sequence_length_symbol,
            'gradients/b_count_10_block::iters': sequence_length_symbol,
            'gradients/b_count_2_block::iters': sequence_length_symbol,
            'gradients/b_count_6_block::iters': sequence_length_symbol,
        }
    else:
        raise NotImplementedError('ERROR: Unknown domain: {}'.format(domain))

    if not is_pytest_run:
        print('\n\nCleaned Graph:\n{}'.format(graph))

    print('\n\nBound values')

    bind_subs = {
        graph_iters_symbol: 1,
        hidden_dim_symbol: base_hidden_dim,
        sequence_length_symbol: base_sequence_length,
        subbatch_size_symbol: base_subbatch_size,
        vocab_size_symbol: base_vocab_size,
        num_workers_symbol: 1,
    }
    var_refs_table = {}
    for var_name, sub_val in bind_str_subs.items():
        var_ref = utils.getIntSymbolFromString(var_name)
        assert var_name not in bind_subs.keys()
        bind_subs[var_ref] = sub_val
        var_refs_table[var_name] = var_ref

    # Verify parameter counts first
    parameters = graph.calcModelParameters()
    if domain == 'wordlm':
        correct_symbolic_params = 16 * hidden_dim_symbol**2 + \
                                  2 * hidden_dim_symbol * vocab_size_symbol + \
                                  8 * hidden_dim_symbol + \
                                  vocab_size_symbol + 1
        correct_params = 104378325
        correct_flops = 2597237004851
        correct_bytes = 144787880084
        correct_total_footprint = 50417112588
    elif domain == 'charlm':
        correct_symbolic_params = 22 * hidden_dim_symbol**2 + \
                                  2 * hidden_dim_symbol * vocab_size_symbol + \
                                  20 * hidden_dim_symbol + \
                                  vocab_size_symbol + 2
        correct_params = 88785316
        correct_flops = 10228050930588
        correct_bytes = 444794005700
        correct_total_footprint = 156135666724
    elif domain == 'nmt':
        correct_symbolic_params = 33 * hidden_dim_symbol**2 + \
                                  3 * hidden_dim_symbol * vocab_size_symbol + \
                                  12 * hidden_dim_symbol + 1
        correct_params = 146890753
        correct_flops = 1053956094363
        correct_bytes = 35992083787
        correct_total_footprint = 14372109278
    else:
        raise NotImplementedError('ERROR: Unknown domain: {}'.format(domain))
    assert sympy.simplify(parameters - correct_symbolic_params) == 0, \
           'Param count incorrect!\n  Expecting: {}\n  Calculated: {}' \
           .format(correct_symbolic_params, parameters)

    print('Symbol associations: {}\n'.format(bind_subs))

    # Calculate model parameter count
    resolved_params = parameters.subs(bind_subs)
    try:
        resolved_params = int(resolved_params)
    except:
        print('ERROR: resolved_params should be int, but is {} = {}'.format(
              type(resolved_params), resolved_params))
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
    assert resolved_bytes == correct_bytes, \
           'Incorrect algorithmic bytes: {}'.format(resolved_bytes)
    print('Alg bytes accessed: {}\nWith specified dims: {}\n'.format(alg_bytes, resolved_bytes))

    # Calculate total memory footprint
    alg_footprint = graph.calcAlgFootprint()
    resolved_footprint = alg_footprint.subs(bind_subs)
    try:
        resolved_footprint = int(resolved_footprint)
    except:
        print('ERROR: resolved_footprint should be int, but is {} = {}'.format(
              type(resolved_footprint), resolved_footprint))
    assert resolved_footprint == correct_total_footprint, \
           'Incorrect algorithmic footprint: {}'.format(resolved_footprint)
    print('Alg mem footprint: {}\nWith specified dims: {}\n'.format(alg_footprint, resolved_footprint))

    # Calculate minimal memory footprint
    alg_min_footprint = graph.calcMinimalFootprint(symbol_subs=bind_subs)
    print('Alg minimal footprint (With specified dims): {}\n'.format(alg_min_footprint))

    # Calculate algorithmic IO per step
    total_io_footprint = 0
    for op in graph.getPlaceholders():
        total_io_footprint += op.calcAlgFootprint()
    if isinstance(total_io_footprint, int):
        resolved_io_footprint = total_io_footprint
    else:
        resolved_io_footprint = total_io_footprint.subs(bind_subs)
    print('Alg IO footprint: {}\nWith specified dims: {}\n'.format(total_io_footprint, resolved_io_footprint))


    if not is_pytest_run:
        print('VERBOSE ALGORTHMIC FLOPS:')
        graph.calcAlgFlops(verbose=True)
        print('')

        print('VERBOSE ALGORTHMIC BYTES:')
        graph.calcAlgBytes(verbose=True)
        print('')

        print('VERBOSE ALGORTHMIC FOOTPRINT:')
        graph.calcAlgFootprint(verbose=True)
        print('')

    # HACKY WAY TO SAVE MODELS FOR NOW!
    pickle.dump(graph, open('cougr/frameworks/example_graphs/tensorflow/full_models/language_models/graph_{}.p'.format(domain), 'wb'))

    print('\n\n======= Algorithmic graph-level analytics: =======')

    if domain == 'wordlm':
        hidden_dims = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 14, 18, 20, 25, 28, 35, 40, 50, 56, 69, 78, 86, 96, 108, 119, 123, 133, 148, 163, 182, 202, 221, 246, 273, 297, 329, 330, 364, 396, 436, 437, 520, 572, 617, 676, 740, 796, 869, 948, 1017, 1106, 1202, 1286, 1394, 1510, 1611, 1742, 1882, 2004, 2161, 2476, 3040, 3714, 4520, 5478, 6628, 8019, 9702, 11739, 14204, 17186, 20795, 25161, 30444, 36837, 38100]
        bind_subs[subbatch_size_symbol] = 128
    elif domain == 'charlm':
        hidden_dims = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 14, 18, 20, 25, 28, 35, 40, 50, 56, 69, 78, 86, 96, 108, 119, 123, 133, 148, 163, 182, 202, 221, 246, 273, 297, 329, 330, 364, 396, 436, 437, 520, 572, 617, 676, 740, 796, 869, 948, 1017, 1106, 1202, 1286, 1394, 1510, 1611, 1742, 1882, 2004, 2161, 2476, 3040, 3714, 5051, 6869, 9341, 12703, 17276, 23495, 31953, 43456, 59100, 80376, 81400]
        bind_subs[subbatch_size_symbol] = 96
    elif domain == 'nmt':
        hidden_dims = [32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072, 3747, 4571, 5576, 6802, 8298, 10123, 12350, 15067, 18381, 22350]
        bind_subs[subbatch_size_symbol] = 96
        bind_subs[sequence_length_symbol] = 26
    else:
        raise NotImplementedError('ERROR: Unknown domain: {}'.format(domain))

    bind_subs.pop(hidden_dim_symbol)
    resolved_params = parameters.subs(bind_subs)

    print('Symbol associations: {}\n'.format(bind_subs))

    print('Algorithmic Flops by hidden dimension, params, and per-batch-sample:')
    resolved_flops = alg_flops.subs(bind_subs)
    for hid_dim in hidden_dims:
        graph_params = resolved_params.subs({hidden_dim_symbol: hid_dim})
        graph_flops = resolved_flops.subs({hidden_dim_symbol: hid_dim})
        graph_flops_per_sample = float(graph_flops) / \
                                 bind_subs[subbatch_size_symbol]
        print('{}\t{}\t{}\t{}'.format(hid_dim, graph_params, graph_flops,
                                      int(graph_flops_per_sample)))

    print('\nAlgorithmic bytes accessed by hidden dimension, params:')
    resolved_bytes = alg_bytes.subs(bind_subs)
    for hid_dim in hidden_dims:
        graph_params = resolved_params.subs({hidden_dim_symbol: hid_dim})
        graph_bytes = resolved_bytes.subs({hidden_dim_symbol: hid_dim})
        print('{}\t{}\t{}'.format(hid_dim, graph_params, graph_bytes))

    print('\nAlgorithmic total memory footprint by hidden dimension, params:')
    resolved_footprint = alg_footprint.subs(bind_subs)
    for hid_dim in hidden_dims:
        graph_params = resolved_params.subs({hidden_dim_symbol: hid_dim})
        graph_footprint = resolved_footprint.subs({hidden_dim_symbol: hid_dim})
        print('{}\t{}\t{}'.format(hid_dim, graph_params, graph_footprint))

    print('\nAlgorithmic minimal memory footprint by hidden dimension, params:')
    full_subs = dict(bind_subs)
    for hid_dim in hidden_dims:
        graph_params = resolved_params.subs({hidden_dim_symbol: hid_dim})
        full_subs[hidden_dim_symbol] = hid_dim
        graph_min_foot = graph.calcMinimalFootprint(symbol_subs=full_subs)
        print('{}\t{}\t{}'.format(hid_dim, graph_params, graph_min_foot))




if False:
    if domain == 'wordlm':
        if args.build_projection:
            # This is hacky anyway... Import required parts here:
            from cougr.ops.optimizer_ops import *
            from cougr.tensors.tensor import *

            projection_dim_symbol = utils.getIntSymbolFromString('projection_dim')
            # (1) Project output of the second recurrent layer. Save the
            # consumers of the output to send the projected values there
            proj_in_op = graph.opsByName['Model/Collapse/Reshape']
            proj_input = proj_in_op._outputs[0]
            proj_input_consumers = proj_input._consumers
            proj_input._consumers = {}
            # (1a) Create projection matrix
            proj_weights = cougr.variable('Model/Collapse/projection/W',
                               [hidden_dim_symbol, projection_dim_symbol], graph)
            # (1b) Create matrix multiply for projection
            proj_mm_out = cougr.matmul('Model/Collapse/projection/MatMul',
                              [None, projection_dim_symbol], proj_input,
                              proj_weights, graph)
            # (2) Feed projection to output consumers
            for op_name, op in proj_input_consumers.items():
                proj_mm_out.consumers[op_name] = op
                for idx in range(len(op._inputs)):
                    if op._inputs[idx] == proj_input:
                        op._inputs[idx] = proj_mm_out
            graph._sinks.pop(proj_mm_out.producer.name)
            # (2a) Fix the softmax dimension
            bind_dict['Model/FullSoftmaxLoss_1/W_Softmax'] = [vocab_size_symbol, projection_dim_symbol]

            # (3) Backprop: Get the tensor for the matmul backprop
            mms_in_op = graph.opsByName['Model/Gradient/Compute/gradients/Model/Collapse/boolean_mask/Reshape_grad/Reshape']
            mms_input = mms_in_op._outputs[0]
            mms_input_consumers = mms_input._consumers
            mms_input._consumers = {}
            print('HEY HERE: {}'.format(mms_in_op.debugString()))

            # (3a) MatMul to backprop to weights
            wt_bp_mm_out = cougr.matmul('Model/Gradient/Compute/gradients/Model/Collapse/projection/MatMul_grad/MatMul',
                               [hidden_dim_symbol, projection_dim_symbol], proj_input, mms_input, graph)
            wt_bp_mm_out.producer.setTransposeInput(0, True)

            # (3b) Apply weights update to proj_weights (in Gradient/Apply?)
            # (3b1) Get inputs to op
            #    Input[0]: proj_weights
            #    Input[1]: Learning rate
            lr_op = graph.opsByName['Model/Gradient/Apply/DistributedGradientDescentOptimizer/learning_rate']
            #    Input[2]: wt_bp_mm_out
            # (3b2) Create op and add to graph (Note: Has no output)
            proj_grad_name = 'Model/Gradient/Apply/DistributedGradientDescentOptimizer/update_Model/projection/W/ApplyGradientDescent'
            proj_grad_op = ApplyGradientDescentOp(proj_grad_name)
            proj_grad_out = Tensor(proj_grad_name, TensorShape(
                                   [hidden_dim_symbol, projection_dim_symbol]))
            proj_grad_op.addOutput(proj_grad_out)
            graph.addOp(proj_grad_op)
            graph.addInputToOp(proj_grad_op, proj_weights)
            graph.addInputToOp(proj_grad_op, lr_op._outputs[0])
            graph.addInputToOp(proj_grad_op, wt_bp_mm_out)

            # (3c) MatMul to backprop to activations
            act_bp_mm_out = cougr.matmul('Model/Gradient/Compute/gradients/Model/Collapse/projection/MatMul_grad/MatMul_1',
                               [subbatch_size_symbol*sequence_length_symbol, hidden_dim_symbol],
                               mms_input, proj_weights, graph)
            act_bp_mm_out.producer.setTransposeInput(1, True)

            # (3d) Connect MatMul activation output to reshape node
            for op_name, op in mms_input_consumers.items():
                act_bp_mm_out.consumers[op_name] = op
                for idx in range(len(op._inputs)):
                    if op._inputs[idx] == mms_input:
                        op._inputs[idx] = act_bp_mm_out
            graph._sinks.pop(act_bp_mm_out.producer.name)

            # (3e) Tie up loose ends:
            allred_op = graph.opsByName['Model/Gradient/Compute/MPIAllreduce_4']
            allred_op._outputs[0].shape.dims[1]._symbol = projection_dim_symbol

            # (4) Propagate shapes to check correctness
            print('Converted to LSTM-p!\n')
            graph.bindTensorShapeDimensions(bind_dict, warn_if_ill_defined=(not is_pytest_run), make_symbolic=True)
            assert graph.isValid()
            print('')
            print(graph)
            lstm_p_params = parameters - hidden_dim_symbol*vocab_size_symbol + (hidden_dim_symbol*projection_dim_symbol + projection_dim_symbol*vocab_size_symbol)
            assert lstm_p_params - graph.calcModelParameters() == 0, \
                'lstm_p_params: {}\ncalcd: {}'.format(lstm_p_params, graph.calcModelParameters())

            # Perform calculations for LSTM-p
            bind_subs[hidden_dim_symbol] = 19968
            bind_subs[projection_dim_symbol] = 2048
            bind_subs[vocab_size_symbol] = 800000
            bind_subs[sequence_length_symbol] = 20
            print('LSTM-p model algorithmic values:')
            # Calculate parameters
            resolved_params = lstm_p_params.subs(bind_subs)
            print('LSTM-p params: {}\nWith specified dims: {}'.format(lstm_p_params, resolved_params))
            # Calculate algorithmic Flops
            lstm_p_flops = graph.calcAlgFlops()
            resolved_flops = lstm_p_flops.subs(bind_subs)
            print('LSTM-p alg_flops: {}\nWith specified dims: {}\n'.format(lstm_p_flops, resolved_flops))
            # Calculate algorthmic Bytes accessed
            lstm_p_bytes = graph.calcAlgBytes()
            resolved_bytes = lstm_p_bytes.subs(bind_subs)
            print('LSTM-p alg_bytes: {}\nWith specified dims: {}\n'.format(lstm_p_bytes, resolved_bytes))
            # Calculate algorthmic Bytes accessed
            lstm_p_foot = graph.calcAlgFootprint()
            resolved_footprint = lstm_p_foot.subs(bind_subs)
            print('LSTM-p alg_foot: {}\nWith specified dims: {}\n'.format(lstm_p_foot, resolved_footprint))
            try: # If the code supports memory footprint estimate, do it
                # Calculate the minimal memory footprint
                lstm_p_min_foot = graph.calcMinFootprint(symbol_subs=bind_subs)
                print('LSTM-p min_foot: {}'.format(lstm_p_min_foot))
                lstm_p_op_intensity = (lstm_p_flops / lstm_p_bytes)
            except:
                pass

            # HACKY WAY TO SAVE MODELS FOR NOW!
            pickle.dump(graph, open('learning_curves_graphs/graph_{}_lstmp.p'.format(domain), 'wb'))

#for hid_dim in hidden_dims:
#    bind_subs[hidden_dim_symbol] = hid_dim
#    graph.calcMinFootprint(symbol_subs=bind_subs)


#for op in graph._ops_by_name.values():
#    op_bytes = op.calcAlgBytes()
#    if op_bytes == 0:
#        continue
#    if isinstance(op_bytes, sympy.Expr):
#        op_bytes_subs = op_bytes.subs(bind_subs)
#    else:
#        op_bytes_subs = op_bytes
#    if op_bytes_subs < 100000000:
#        continue
#    print('{}: {} = {}'.format(op.name, op_bytes, op_bytes_subs))


#outfile = 'for_newsha.txt'
#ofhandle = open(outfile, 'w')
#for op in graph._ops_by_name.values():
#    op_flops = op.calcAlgFlops()
#    if isinstance(op_flops, sympy.Expr):
#        op_flops = op_flops.subs(bind_subs)
#    op_bytes = op.calcAlgBytes()
#    if isinstance(op_bytes, sympy.Expr):
#        op_bytes = op_bytes.subs(bind_subs)
#    op_foot = op.calcAlgFootprint()
#    if isinstance(op_foot, sympy.Expr):
#        op_foot = op_foot.subs(bind_subs)
#    junk = ofhandle.write('Op: {} of type {}\n    flops: {}\n    bytes: {}\n    footprint: {}\n'.format(op.name, type(op), op_flops, op_bytes, op_foot))
#
#ofhandle.close()


#bind_final_subs = { hidden_dim_symbol: 1, subbatch_size_symbol: base_subbatch_size, sequence_length_symbol: base_sequence_length }
#
#for op_name in sorted(graph._ops_by_name.keys()):
#    op = graph._ops_by_name[op_name]
#    op_flops = op.calcAlgFlops()
#    if op_flops == 0:
#        continue
#    if isinstance(op_flops, sympy.Expr):
#        op_flops = op_flops.subs(bind_subs)
#        op_flops_bound = op_flops.subs(bind_final_subs)
#    else:
#        op_flops_bound = op_flops
#    if op_flops_bound < 1000000:
#        continue
#    print('{}: {} = {}'.format(op.name, op_flops, op_flops_bound))


if __name__ == "__main__":
    domain_choices = ['wordlm', 'charlm', 'nmt']
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', choices=domain_choices, required=True,
                        help='The domain to test ({})'.format(domain_choices))
    parser.add_argument('--build_projection', action="store_true",
                    help='Add a projection layer to the model')
    args = parser.parse_args()

    run_tf_language_model(domain=args.domain,
                          build_projection=args.build_projection)
