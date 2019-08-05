import numpy as np
import pickle
import sympy
import sys
sys.setrecursionlimit(50000)


from catamount.api import utils
import catamount.frameworks.tensorflow
from catamount.ops.constant import *
from catamount.ops.unknown_op import UnknownOp
from catamount.ops.variable import *


is_pytest_run = False

def test_tf_speech_attention():
    global is_pytest_run
    is_pytest_run = True
    run_tf_speech_attention()

def run_tf_speech_attention():
    global is_pytest_run
    graph_meta = 'catamount/frameworks/example_graphs/tensorflow/full_models/speech_attention/model.ckpt.meta'

    graph = catamount.frameworks.tensorflow.import_graph(graph_meta)
    assert graph.isValid()

    # HAX: NEED TO MANUALLY REMOVE SOME?! WHY?
    remove_ops = ['DevArgmaxWERChecker/Less', 'DevLossChecker/Less', 'DevArgmaxWERChecker/best_dev', 'DevLossChecker/best_dev']
    for op_name in remove_ops:
        op = graph.opsByName[op_name]
        graph.removeOp(op)
    assert graph.isValid()

    # Remove ops that are not executed during a standard training step:
    graph_ops = list(graph._ops_by_name.values())
    for op in graph_ops:
         # Ops in attn_model_[1-3] are used for inference
         if 'attn_model_1' in op.name or \
            'attn_model_2' in op.name or \
            'attn_model_3' in op.name:
             graph.removeOp(op)
    assert graph.isValid()


    print('Initial graph:\n{}\n'.format(graph))
    init_params = graph.calcModelParameters()
    print('Initial parameters: {}'.format(init_params))
    print('Initial Flops: {}\n'.format(graph.calcAlgFlops()))

    print('Placeholders:')
    for op in graph.getPlaceholders():
        print(op.debugString())
    print('')

    # Set up symbols to name dimensions
    audio_features_symbol = utils.getPositiveIntSymbolFromString('audio_features')
    encoder_steps_symbol = utils.getPositiveIntSymbolFromString('encoder_steps')
    decoder_steps_symbol = utils.getPositiveIntSymbolFromString('decoder_steps')
    subbatch_size_symbol = utils.getPositiveIntSymbolFromString('subbatch_size')
    attn_dim_symbol = utils.getPositiveIntSymbolFromString('attn_dim')
    attn_hidden_dim_symbol = utils.getPositiveIntSymbolFromString('attn_hidden_dim')
    dec_hidden_dim_symbol = utils.getPositiveIntSymbolFromString('dec_hidden_dim')
    enc_hidden_dim_symbol = utils.getPositiveIntSymbolFromString('enc_hidden_dim')
    graph_iters_symbol = utils.getIntSymbolFromString('graph::iters')
    output_vocab_symbol = utils.getPositiveIntSymbolFromString('output_vocab')
    conv_width_symbol = utils.getPositiveIntSymbolFromString('conv_width')
    num_conv_filters_symbol = utils.getPositiveIntSymbolFromString('num_conv_filters')

    # Convert these constant dimensions to symbols
    base_encoder_steps = 300
    base_decoder_steps = 300
    base_subbatch_size = 32
    base_output_vocab = 31
    base_audio_features = 40
    base_conv_width = 53
    base_attn_dim = 137
    base_attn_hidden_dim = 509
    base_dec_hidden_dim = 571
    base_enc_hidden_dim = 1051
    base_enc_input_dim = 1091 # Input + recurrent state
    enc_input_dim_symbol = audio_features_symbol + enc_hidden_dim_symbol
    base_dec_attn_rec = 2133
    dec_attn_rec_symbol = 2 * enc_hidden_dim_symbol + output_vocab_symbol
    base_attn_cell_inputs = 2611
    attn_cell_inputs_symbol = 2 * enc_hidden_dim_symbol + attn_hidden_dim_symbol
    base_attn_cell_in_dim = 2642
    attn_cell_in_dim_symbol = 2 * enc_hidden_dim_symbol + output_vocab_symbol + \
                              attn_hidden_dim_symbol
    base_dec_attn_dim = 3182
    dec_attn_dim_symbol = attn_hidden_dim_symbol + 2 * enc_hidden_dim_symbol + \
                          dec_hidden_dim_symbol

    bind_dict = { # Placeholders
                  'attn_model/input_seq': [encoder_steps_symbol, subbatch_size_symbol, audio_features_symbol],
                  'attn_model/input_len': [subbatch_size_symbol],
                  'attn_model/output_seq': [decoder_steps_symbol, subbatch_size_symbol],
                  'attn_model/output_mask': [decoder_steps_symbol, subbatch_size_symbol],

                  # Variables
                  'InputNormalizer/means': [audio_features_symbol],
                  'InputNormalizer/std': [audio_features_symbol],
                  'attn_model/AffineAttentionStateNN/W': [2 * enc_hidden_dim_symbol, attn_dim_symbol],
                  'attn_model/AffineAttentionStateNN/b': [attn_dim_symbol],
                  'attn_model/AffineOutputProjection/W': [dec_hidden_dim_symbol, output_vocab_symbol],
                  'attn_model/AffineOutputProjection/b': [output_vocab_symbol],
                  'attn_model/Decoder/attn_model/attention_cell/biases': [4 * attn_hidden_dim_symbol],
                  'attn_model/Decoder/attn_model/attention_cell/weights': [attn_hidden_dim_symbol + 2 * enc_hidden_dim_symbol + output_vocab_symbol, 4 * attn_hidden_dim_symbol],
                  'attn_model/Decoder/attn_model/decoder_cell/biases': [4 * dec_hidden_dim_symbol],
                  'attn_model/Decoder/attn_model/decoder_cell/weights': [attn_hidden_dim_symbol + dec_hidden_dim_symbol + 2 * enc_hidden_dim_symbol, 4 * dec_hidden_dim_symbol],
                  'attn_model/HybridAttentionContext/Q': [conv_width_symbol, 1, num_conv_filters_symbol],
                  'attn_model/HybridAttentionContext/U': [1, num_conv_filters_symbol, attn_dim_symbol],
                  'attn_model/HybridAttentionContext/W': [2 * attn_hidden_dim_symbol, attn_dim_symbol],
                  'attn_model/HybridAttentionContext/b': [attn_dim_symbol],
                  'attn_model/HybridAttentionContext/w': [attn_dim_symbol],
                  'attn_model/StackedEncoder/Layer0/RNNEncoder/bidirectional_rnn/bw/basic_lstm_cell/bias': [4 * enc_hidden_dim_symbol],
                  'attn_model/StackedEncoder/Layer0/RNNEncoder/bidirectional_rnn/bw/basic_lstm_cell/kernel': [audio_features_symbol + enc_hidden_dim_symbol, 4 * enc_hidden_dim_symbol],
                  'attn_model/StackedEncoder/Layer0/RNNEncoder/bidirectional_rnn/fw/basic_lstm_cell/bias': [4 * enc_hidden_dim_symbol],
                  'attn_model/StackedEncoder/Layer0/RNNEncoder/bidirectional_rnn/fw/basic_lstm_cell/kernel': [audio_features_symbol + enc_hidden_dim_symbol, 4 * enc_hidden_dim_symbol],
                  'attn_model/StackedEncoder/Layer2/RNNEncoder/bidirectional_rnn/bw/basic_lstm_cell/bias': [4 * enc_hidden_dim_symbol],
                  'attn_model/StackedEncoder/Layer2/RNNEncoder/bidirectional_rnn/bw/basic_lstm_cell/kernel': [3 * enc_hidden_dim_symbol, 4 * enc_hidden_dim_symbol],
                  'attn_model/StackedEncoder/Layer2/RNNEncoder/bidirectional_rnn/fw/basic_lstm_cell/bias': [4 * enc_hidden_dim_symbol],
                  'attn_model/StackedEncoder/Layer2/RNNEncoder/bidirectional_rnn/fw/basic_lstm_cell/kernel': [3 * enc_hidden_dim_symbol, 4 * enc_hidden_dim_symbol],
                  'attn_model/StackedEncoder/Layer4/RNNEncoder/bidirectional_rnn/bw/basic_lstm_cell/bias': [4 * enc_hidden_dim_symbol],
                  'attn_model/StackedEncoder/Layer4/RNNEncoder/bidirectional_rnn/bw/basic_lstm_cell/kernel': [3 * enc_hidden_dim_symbol, 4 * enc_hidden_dim_symbol],
                  'attn_model/StackedEncoder/Layer4/RNNEncoder/bidirectional_rnn/fw/basic_lstm_cell/bias': [4 * enc_hidden_dim_symbol],
                  'attn_model/StackedEncoder/Layer4/RNNEncoder/bidirectional_rnn/fw/basic_lstm_cell/kernel': [3 * enc_hidden_dim_symbol, 4 * enc_hidden_dim_symbol],

                  # Constants
                  'attn_model/AttentionModel/gradients/attn_model/Decoder/while/MatMul/Enter_grad/b_acc': [dec_hidden_dim_symbol, output_vocab_symbol],
                  'attn_model/AttentionModel/gradients/attn_model/Decoder/while/add/Enter_grad/b_acc': [output_vocab_symbol],
                  'attn_model/AttentionModel/gradients/attn_model/Decoder/while/attn_model/MatMul/Enter_grad/b_acc': [2 * attn_hidden_dim_symbol, attn_dim_symbol],
                  'attn_model/AttentionModel/gradients/attn_model/Decoder/while/attn_model/add_2/Enter_grad/b_acc': [attn_dim_symbol],
                  'attn_model/AttentionModel/gradients/attn_model/Decoder/while/attn_model/attention_cell/BiasAdd/Enter_grad/b_acc': [4 * attn_hidden_dim_symbol],
                  'attn_model/AttentionModel/gradients/attn_model/Decoder/while/attn_model/attention_cell/attention_cell/add/Enter_grad/b_acc': [attn_hidden_dim_symbol + 2 * enc_hidden_dim_symbol + output_vocab_symbol, 4 * attn_hidden_dim_symbol],
                  'attn_model/AttentionModel/gradients/attn_model/Decoder/while/attn_model/conv1d/ExpandDims_1/Enter_grad/b_acc': [conv_width_symbol, 1, 4],
                  'attn_model/AttentionModel/gradients/attn_model/Decoder/while/attn_model/conv1d_1/ExpandDims_1/Enter_grad/b_acc': [1, 4, attn_dim_symbol],
                  'attn_model/AttentionModel/gradients/attn_model/Decoder/while/attn_model/decoder_cell/BiasAdd/Enter_grad/b_acc': [4 * dec_hidden_dim_symbol],
                  'attn_model/AttentionModel/gradients/attn_model/Decoder/while/attn_model/decoder_cell/decoder_cell/add/Enter_grad/b_acc': [attn_hidden_dim_symbol + dec_hidden_dim_symbol + 2 * enc_hidden_dim_symbol, 4 * dec_hidden_dim_symbol],
                  'attn_model/AttentionModel/gradients/attn_model/Decoder/while/attn_model/mul/Enter_grad/b_acc': [attn_dim_symbol],
                  'attn_model/AttentionModel/gradients/attn_model/StackedEncoder/Layer0/RNNEncoder/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc': [4 * enc_hidden_dim_symbol],
                  'attn_model/AttentionModel/gradients/attn_model/StackedEncoder/Layer0/RNNEncoder/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc': [audio_features_symbol + enc_hidden_dim_symbol, 4 * enc_hidden_dim_symbol],
                  'attn_model/AttentionModel/gradients/attn_model/StackedEncoder/Layer0/RNNEncoder/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc': [4 * enc_hidden_dim_symbol],
                  'attn_model/AttentionModel/gradients/attn_model/StackedEncoder/Layer0/RNNEncoder/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc': [audio_features_symbol + enc_hidden_dim_symbol, 4 * enc_hidden_dim_symbol],
                  'attn_model/AttentionModel/gradients/attn_model/StackedEncoder/Layer2/RNNEncoder/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc': [4 * enc_hidden_dim_symbol],
                  'attn_model/AttentionModel/gradients/attn_model/StackedEncoder/Layer2/RNNEncoder/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc': [3 * enc_hidden_dim_symbol, 4 * enc_hidden_dim_symbol],
                  'attn_model/AttentionModel/gradients/attn_model/StackedEncoder/Layer2/RNNEncoder/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc': [4 * enc_hidden_dim_symbol],
                  'attn_model/AttentionModel/gradients/attn_model/StackedEncoder/Layer2/RNNEncoder/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc': [3 * enc_hidden_dim_symbol, 4 * enc_hidden_dim_symbol],
                  'attn_model/AttentionModel/gradients/attn_model/StackedEncoder/Layer4/RNNEncoder/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc': [4 * enc_hidden_dim_symbol],
                  'attn_model/AttentionModel/gradients/attn_model/StackedEncoder/Layer4/RNNEncoder/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc': [3 * enc_hidden_dim_symbol, 4 * enc_hidden_dim_symbol],
                  'attn_model/AttentionModel/gradients/attn_model/StackedEncoder/Layer4/RNNEncoder/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc': [4 * enc_hidden_dim_symbol],
                  'attn_model/AttentionModel/gradients/attn_model/StackedEncoder/Layer4/RNNEncoder/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter_grad/b_acc': [3 * enc_hidden_dim_symbol, 4 * enc_hidden_dim_symbol],
                }

    # Update constant values
    const_dict = {
                   'attn_model/AffineAttentionStateNN/Reshape/shape': [-1, 2 * enc_hidden_dim_symbol],
                   'attn_model/AffineAttentionStateNN/Reshape_1/shape/2': attn_dim_symbol,
                   'attn_model/AttentionEncoderDecoder/Reshape/shape/1': output_vocab_symbol,
                   'attn_model/AttentionModel/gradients/attn_model/AffineAttentionStateNN/add_grad/Shape_1': [attn_dim_symbol],
                   'attn_model/AttentionModel/gradients/attn_model/Decoder/while/add_grad/Shape_1': [output_vocab_symbol],
                   'attn_model/AttentionModel/gradients/attn_model/Decoder/while/attn_model/add_2_grad/Shape_1': [attn_dim_symbol],
                   'attn_model/AttentionModel/gradients/attn_model/Decoder/while/attn_model/conv1d/Conv2D_grad/Const': [1, conv_width_symbol, 1, num_conv_filters_symbol],
                   'attn_model/AttentionModel/gradients/attn_model/Decoder/while/attn_model/conv1d/ExpandDims_1_grad/Shape': [conv_width_symbol, 1, num_conv_filters_symbol],
                   'attn_model/AttentionModel/gradients/attn_model/Decoder/while/attn_model/conv1d_1/Conv2D_grad/Const': [1, 1, num_conv_filters_symbol, attn_dim_symbol],
                   'attn_model/AttentionModel/gradients/attn_model/Decoder/while/attn_model/conv1d_1/ExpandDims_1_grad/Shape': [1, num_conv_filters_symbol, attn_dim_symbol],
                   'attn_model/AttentionModel/gradients/attn_model/Decoder/while/attn_model/mul_grad/Shape_1': [attn_dim_symbol],
                   'attn_model/Decoder/CustomLSTMCellZeroState/Const': [2 * attn_hidden_dim_symbol],
                   'attn_model/Decoder/CustomLSTMCellZeroState/Const_1': [2 * attn_hidden_dim_symbol],
                   'attn_model/Decoder/CustomLSTMCellZeroState_1/Const': [2 * dec_hidden_dim_symbol],
                   'attn_model/Decoder/CustomLSTMCellZeroState_1/Const_1': [2 * dec_hidden_dim_symbol],
                   'attn_model/Decoder/while/attn_model/attention_cell/attention_cell/Shape': [attn_hidden_dim_symbol + 2 * enc_hidden_dim_symbol + output_vocab_symbol, 4 * attn_hidden_dim_symbol],
                   'attn_model/Decoder/while/attn_model/decoder_cell/decoder_cell/Shape': [attn_hidden_dim_symbol + dec_hidden_dim_symbol + 2 * enc_hidden_dim_symbol, 4 * dec_hidden_dim_symbol],
                   'attn_model/Decoder/while/attn_model/one_hot/depth': output_vocab_symbol,
                   'attn_model/Decoder/zeros/shape/1': 2 * enc_hidden_dim_symbol,
                   'attn_model/Decoder/zeros_2/shape/1': output_vocab_symbol,
                   'attn_model/Reshape/shape': [1, 1, audio_features_symbol],
                   'attn_model/Reshape_1/shape': [1, 1, audio_features_symbol],
                   'attn_model/Reshape_2/shape/2': 2 * enc_hidden_dim_symbol,
                   'attn_model/StackedEncoder/Layer0/RNNEncoder/Reshape/shape/2': audio_features_symbol,
                   'attn_model/StackedEncoder/Layer0/RNNEncoder/bidirectional_rnn/bw/bw/BasicLSTMCellZeroState/Const': [2 * enc_hidden_dim_symbol],
                   'attn_model/StackedEncoder/Layer0/RNNEncoder/bidirectional_rnn/bw/bw/BasicLSTMCellZeroState/Const_1': [2 * enc_hidden_dim_symbol],
                   'attn_model/StackedEncoder/Layer0/RNNEncoder/bidirectional_rnn/bw/bw/Const_1': [enc_hidden_dim_symbol],
                   'attn_model/StackedEncoder/Layer0/RNNEncoder/bidirectional_rnn/bw/bw/Const_4': [enc_hidden_dim_symbol],
                   'attn_model/StackedEncoder/Layer0/RNNEncoder/bidirectional_rnn/fw/fw/BasicLSTMCellZeroState/Const': [2 * enc_hidden_dim_symbol],
                   'attn_model/StackedEncoder/Layer0/RNNEncoder/bidirectional_rnn/fw/fw/BasicLSTMCellZeroState/Const_1': [2 * enc_hidden_dim_symbol],
                   'attn_model/StackedEncoder/Layer0/RNNEncoder/bidirectional_rnn/fw/fw/Const_1': [enc_hidden_dim_symbol],
                   'attn_model/StackedEncoder/Layer0/RNNEncoder/bidirectional_rnn/fw/fw/Const_4': [enc_hidden_dim_symbol],
                   'attn_model/StackedEncoder/Layer2/RNNEncoder/Reshape/shape/2': 2 * enc_hidden_dim_symbol,
                   'attn_model/StackedEncoder/Layer2/RNNEncoder/bidirectional_rnn/bw/bw/BasicLSTMCellZeroState/Const': [2 * enc_hidden_dim_symbol],
                   'attn_model/StackedEncoder/Layer2/RNNEncoder/bidirectional_rnn/bw/bw/BasicLSTMCellZeroState/Const_1': [2 * enc_hidden_dim_symbol],
                   'attn_model/StackedEncoder/Layer2/RNNEncoder/bidirectional_rnn/bw/bw/Const_1': [enc_hidden_dim_symbol],
                   'attn_model/StackedEncoder/Layer2/RNNEncoder/bidirectional_rnn/bw/bw/Const_4': [enc_hidden_dim_symbol],
                   'attn_model/StackedEncoder/Layer2/RNNEncoder/bidirectional_rnn/fw/fw/BasicLSTMCellZeroState/Const': [2 * enc_hidden_dim_symbol],
                   'attn_model/StackedEncoder/Layer2/RNNEncoder/bidirectional_rnn/fw/fw/BasicLSTMCellZeroState/Const_1': [2 * enc_hidden_dim_symbol],
                   'attn_model/StackedEncoder/Layer2/RNNEncoder/bidirectional_rnn/fw/fw/Const_1': [enc_hidden_dim_symbol],
                   'attn_model/StackedEncoder/Layer2/RNNEncoder/bidirectional_rnn/fw/fw/Const_4': [enc_hidden_dim_symbol],
                   'attn_model/StackedEncoder/Layer4/RNNEncoder/Reshape/shape/2': 2 * enc_hidden_dim_symbol,
                   'attn_model/StackedEncoder/Layer4/RNNEncoder/bidirectional_rnn/bw/bw/BasicLSTMCellZeroState/Const': [2 * enc_hidden_dim_symbol],
                   'attn_model/StackedEncoder/Layer4/RNNEncoder/bidirectional_rnn/bw/bw/BasicLSTMCellZeroState/Const_1': [2 * enc_hidden_dim_symbol],
                   'attn_model/StackedEncoder/Layer4/RNNEncoder/bidirectional_rnn/bw/bw/Const_1': [enc_hidden_dim_symbol],
                   'attn_model/StackedEncoder/Layer4/RNNEncoder/bidirectional_rnn/bw/bw/Const_4': [enc_hidden_dim_symbol],
                   'attn_model/StackedEncoder/Layer4/RNNEncoder/bidirectional_rnn/fw/fw/BasicLSTMCellZeroState/Const': [2 * enc_hidden_dim_symbol],
                   'attn_model/StackedEncoder/Layer4/RNNEncoder/bidirectional_rnn/fw/fw/BasicLSTMCellZeroState/Const_1': [2 * enc_hidden_dim_symbol],
                   'attn_model/StackedEncoder/Layer4/RNNEncoder/bidirectional_rnn/fw/fw/Const_1': [enc_hidden_dim_symbol],
                   'attn_model/StackedEncoder/Layer4/RNNEncoder/bidirectional_rnn/fw/fw/Const_4': [enc_hidden_dim_symbol],
                 }

    graph.bindConstantValues(const_dict)

    # TODO: Currently, Catamount doesn't automatically handle Tensorflow TensorArrays
    # or Stack ops. Here, manually set the dimensions of these ops' tensors.
    for op in graph._ops_by_name.values():
        op_name_suffix = op.name.split('/')[-1]
        if 'TensorArrayGather' in op_name_suffix:
            assert isinstance(op, UnknownOp)
            assert len(op._inputs) == 3
            assert len(op._outputs) == 1
            if op._outputs[0].shape.rank == 1 or op._outputs[0].shape.rank == 2:
                if len(op._outputs[0].consumers) > 0:
                    print('TODO: Unknown TensorArrayGather (rank {}): {}'
                          .format(op._outputs[0].shape.rank, op.debugString()))
            elif op._outputs[0].shape.isUnknown() or op._outputs[0].shape.rank == 3:
                if len(op._outputs[0].consumers) > 0:
                    # If output rank is 3, then appears to be:
                    # [seq_length, batch_size, enc_hid], where
                    # seq_length depends on layer
                    out_shape = None
                    if 'StackedEncoder/Layer0' in op.name:
                        out_shape = [encoder_steps_symbol,
                                     subbatch_size_symbol,
                                     enc_hidden_dim_symbol]
                    elif 'StackedEncoder/Layer2' in op.name:
                        if 'attn_model/AttentionModel/gradients' in op.name:
                            # Backprop stores concatenated state
                            out_shape = [encoder_steps_symbol // 2,
                                         subbatch_size_symbol,
                                         2 * enc_hidden_dim_symbol]
                        else:
                            out_shape = [encoder_steps_symbol // 2,
                                         subbatch_size_symbol,
                                         enc_hidden_dim_symbol]
                    elif 'StackedEncoder/Layer4' in op.name:
                        if 'attn_model/AttentionModel/gradients' in op.name:
                            # Backprop stores concatenated state
                            out_shape = [(encoder_steps_symbol // 2) // 2,
                                         subbatch_size_symbol,
                                         2 * enc_hidden_dim_symbol]
                        else:
                            out_shape = [(encoder_steps_symbol // 2) // 2,
                                         subbatch_size_symbol,
                                         enc_hidden_dim_symbol]
                    elif 'Decoder' in op.name:
                        # HAXXXX: Manually specify a few
                        if op.name == 'attn_model/Decoder/TensorArrayStack/TensorArrayGatherV3':
                            out_shape = [decoder_steps_symbol,
                                         subbatch_size_symbol,
                                         output_vocab_symbol]
                        else:
                            out_shape = [decoder_steps_symbol,
                                         subbatch_size_symbol,
                                         dec_hidden_dim_symbol]
                    else:
                        print('TODO: Unknown TensorArrayGather {}'
                              .format(op.debugString()))
                    if out_shape is not None:
                        op._outputs[0].mergeShape(out_shape, make_symbolic=True)
            else:
                print('TODO: Unknown TensorArrayGather {}'
                      .format(op.debugString()))
        elif 'TensorArraySize' in op_name_suffix:
            assert isinstance(op, UnknownOp)
            assert len(op._inputs) == 2
            assert len(op._outputs) == 1
            assert op._outputs[0].shape.rank == 0
            # NOTES:
            # StackedEncoder Layer0: enc_seq
            # StackedEncoder Layer2: enc_seq / 2 # Due to stride 2 in time
            # StackedEncoder Layer4: enc_seq / 4 # Due to stride 2 in time
            # Decoder: dec_seq
            if 'StackedEncoder/Layer0' in op.name:
                op._outputs[0].setValue(encoder_steps_symbol)
            elif 'StackedEncoder/Layer2' in op.name:
                op._outputs[0].setValue(encoder_steps_symbol // 2)
            elif 'StackedEncoder/Layer4' in op.name:
                op._outputs[0].setValue((encoder_steps_symbol // 2) // 2)
            elif 'Decoder' in op.name:
                op._outputs[0].setValue(decoder_steps_symbol)
            else:
                print('WARN: Unknown TensorArraySizeV3: {}'
                      .format(op.debugString()))
        elif 'TensorArrayRead' in op_name_suffix:
            assert isinstance(op, UnknownOp)
            assert len(op._inputs) == 3
            assert len(op._outputs) == 1
            assert op._outputs[0].shape.isUnknown() or \
                   op._outputs[0].shape.rank == 2, \
                   '{}'.format(op.name)
            if op._outputs[0].shape.isUnknown():
                if len(op._outputs[0].consumers) > 0:
                    out_shape = None
                    if 'attn_model/AttentionModel/gradients/attn_model/StackedEncoder/Layer' in op.name and \
                       ('/RNNEncoder/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3' in op.name or \
                        '/RNNEncoder/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3' in op.name):
                        out_shape = [subbatch_size_symbol,
                                     enc_hidden_dim_symbol]
                    elif op.name == 'attn_model/AttentionModel/gradients/attn_model/Decoder/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3' or \
                         op.name == 'attn_model/AttentionModel/gradients/attn_model/Decoder/while/TensorArrayWrite_1/TensorArrayWriteV3_grad/TensorArrayReadV3' or \
                         op.name == 'attn_model_2/Decoder/while/cond/TensorArrayReadV3' or \
                         op.name == 'attn_model/Decoder/while/cond/TensorArrayReadV3':
                        out_shape = [subbatch_size_symbol,
                                     output_vocab_symbol]
                    else:
                        print('WARN: Unknown TensorArrayReadV3 out shape: {}'
                              .format(op.debugString()))
                    if out_shape is not None:
                        op._outputs[0].mergeShape(out_shape, make_symbolic=True)
            else:
                # NOTES: Many are (?, 40 "features"), (?, 1051 "enc_hid"), or (?, 2102 "2*enc_hid")
                dim_1_val = op._outputs[0].shape.getDimension(1).value
                assert dim_1_val == base_audio_features or \
                       dim_1_val == base_enc_hidden_dim or \
                       dim_1_val == 2 * base_enc_hidden_dim, \
                       'Op: {}\n   Dim 1 value: {}'.format(op.debugString(), dim_1_val)
                out_shape = None
                if dim_1_val == base_audio_features:
                    out_shape = [subbatch_size_symbol, audio_features_symbol]
                elif dim_1_val > 0 and dim_1_val % base_enc_hidden_dim == 0:
                    mult = dim_1_val // base_enc_hidden_dim
                    out_shape = [subbatch_size_symbol, mult * enc_hidden_dim_symbol]
                else:
                    print('Unhandled TensorArrayRead: {}'.format(op.debugString()))
                if out_shape is not None:
                    op._outputs[0].mergeShape(out_shape, make_symbolic=True)

    # Manually set a couple shapes for max ops that can't yet resolve
    # maximums of 1 vs. positive symbols:
    max_op = graph._ops_by_name['attn_model/AttentionModel/gradients/attn_model/AttentionEncoderDecoder/Sum_grad/Maximum']
    max_op._outputs[0].mergeShape([2])
    max_op._outputs[0].setValue([1, subbatch_size_symbol])

    max_op = graph._ops_by_name['attn_model/AttentionModel/gradients/attn_model/Decoder/while/attn_model/Sum_grad/Maximum']
    max_op._outputs[0].mergeShape([3])
    # [floor(floor(encoder_steps/2)/2) subbatch_size 1]
    max_op._outputs[0].setValue([(encoder_steps_symbol // 2) // 2,
                                 subbatch_size_symbol, 1])

    max_op = graph._ops_by_name['attn_model/AttentionModel/gradients/attn_model/Decoder/while/attn_model/Sum_1_grad/Maximum']
    max_op._outputs[0].mergeShape([3])
    # [1 subbatch_size 2*enc_hidden_dim]
    max_op._outputs[0].setValue([1, subbatch_size_symbol,
                                 2 * enc_hidden_dim_symbol])

    print('Binding variables')

    graph.bindShapesAndPropagate(bind_dict, warn_if_ill_defined=(not is_pytest_run), make_symbolic=True)
    assert graph.isValid()

    print('\n\nCleaned Graph:\n{}'.format(graph))

    print('\n\nBound values')

    # Set base values to be subbed in:
    base_encoder_steps = 96
    base_decoder_steps = 24
    base_attn_dim = 128
    base_conv_width = 50
    base_attn_hidden_dim = 512
    base_dec_hidden_dim = 512
    base_enc_hidden_dim = 1024

    bind_subs = { audio_features_symbol: base_audio_features,
                  encoder_steps_symbol: base_encoder_steps,
                  decoder_steps_symbol: (encoder_steps_symbol // 2) // 2,
                  subbatch_size_symbol: base_subbatch_size,
                  attn_dim_symbol: base_attn_dim,
                  attn_hidden_dim_symbol: enc_hidden_dim_symbol // 2,
                  dec_hidden_dim_symbol: enc_hidden_dim_symbol // 2,
                  output_vocab_symbol: base_output_vocab,
                  conv_width_symbol: base_conv_width,
                  enc_hidden_dim_symbol: base_enc_hidden_dim,
                  num_conv_filters_symbol: 4,
                  graph_iters_symbol: 1,
                }
    # Add loop iteration counts to bind_subs
    bind_str_subs = {
        'attn_model/AttentionModel/gradients/b_count_2_block::iters': decoder_steps_symbol,
        'attn_model/Decoder/while/LoopCond_block::iters': decoder_steps_symbol,
        'attn_model/AttentionModel/gradients/b_count_22_block::iters': encoder_steps_symbol,
        'attn_model/AttentionModel/gradients/b_count_26_block::iters': encoder_steps_symbol,
        'attn_model/StackedEncoder/Layer0/RNNEncoder/bidirectional_rnn/bw/bw/while/LoopCond_block::iters': encoder_steps_symbol,
        'attn_model/StackedEncoder/Layer0/RNNEncoder/bidirectional_rnn/fw/fw/while/LoopCond_block::iters': encoder_steps_symbol,
        'attn_model/AttentionModel/gradients/b_count_14_block::iters': encoder_steps_symbol // 2,
        'attn_model/AttentionModel/gradients/b_count_18_block::iters': encoder_steps_symbol // 2,
        'attn_model/StackedEncoder/Layer2/RNNEncoder/bidirectional_rnn/bw/bw/while/LoopCond_block::iters': encoder_steps_symbol // 2,
        'attn_model/StackedEncoder/Layer2/RNNEncoder/bidirectional_rnn/fw/fw/while/LoopCond_block::iters': encoder_steps_symbol // 2,
        'attn_model/AttentionModel/gradients/b_count_6_block::iters': (encoder_steps_symbol // 2) // 2,
        'attn_model/AttentionModel/gradients/b_count_10_block::iters': (encoder_steps_symbol // 2) // 2,
        'attn_model/StackedEncoder/Layer4/RNNEncoder/bidirectional_rnn/bw/bw/while/LoopCond_block::iters': (encoder_steps_symbol // 2) // 2,
        'attn_model/StackedEncoder/Layer4/RNNEncoder/bidirectional_rnn/fw/fw/while/LoopCond_block::iters': (encoder_steps_symbol // 2) // 2,
        }

    for var_name, sub_val in bind_str_subs.items():
        var_ref = utils.getIntSymbolFromString(var_name)
        assert var_name not in bind_subs.keys()
        bind_subs[var_ref] = sub_val

    # Calculate model parameter count
    parameters = graph.calcModelParameters()
    resolved_params = parameters.subs(bind_subs)
    try:
        resolved_params = int(resolved_params)
    except:
        print('ERROR: resolved_params should be int, but is {} = {}'.format(
              type(resolved_params), resolved_params))
    correct_params = 71084729
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
    correct_flops = 568878183032
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
    correct_bytes = 92231419797
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
    correct_footprint = 32624988214
    assert resolved_footprint == correct_footprint, \
           'Incorrect algorithmic footprint: {}'.format(resolved_footprint)
    print('Alg mem footprint: {}\nWith specified dims: {}\n'.format(alg_footprint, resolved_footprint))

    # Calculate algorithmic IO per step
    total_io_footprint = 0
    for op in graph.getPlaceholders():
        total_io_footprint += op.calcAlgFootprint()
    resolved_io_footprint = total_io_footprint.subs(bind_subs)
    print('Alg IO footprint: {}\nWith specified dims: {}\n'.format(total_io_footprint, resolved_io_footprint))


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
    pickle.dump(graph, open('catamount/frameworks/example_graphs/tensorflow/full_models/speech_attention/graph_speech_attention.p', 'wb'))


    if is_pytest_run:
        return

    print('\n\n======= Algorithmic graph-level analytics: =======')

    encoder_dims = [32, 64, 96, 128, 160, 192, 256, 320, 384, 448, 512, 640, 768, 892, 1024, 1152, 1280, 1408, 1548, 1702, 1872, 2059, 2264, 2490, 2739, 3012, 3289]
    base_encoder_steps = 335
    base_subbatch_size = 32
    base_attn_dim = 128
    base_conv_width = 50
    base_attn_hidden_dim = 512
    base_dec_hidden_dim = 512
    base_enc_hidden_dim = 1024

    bind_subs[audio_features_symbol] = base_audio_features
    bind_subs[encoder_steps_symbol] = base_encoder_steps
    bind_subs[decoder_steps_symbol] = (encoder_steps_symbol // 2) // 2
    bind_subs[subbatch_size_symbol] = base_subbatch_size
    bind_subs[attn_dim_symbol] = base_attn_dim
    bind_subs[attn_hidden_dim_symbol] = enc_hidden_dim_symbol // 2
    bind_subs[dec_hidden_dim_symbol] = enc_hidden_dim_symbol // 2
    bind_subs[output_vocab_symbol] = base_output_vocab
    bind_subs[conv_width_symbol] = base_conv_width
    # bind_subs[enc_hidden_dim_symbol] = base_enc_hidden_dim
    bind_subs[num_conv_filters_symbol] = 4
    bind_subs[graph_iters_symbol] = 1

    bind_subs.pop(enc_hidden_dim_symbol)
    resolved_params = parameters.subs(bind_subs)

    print('Symbol associations: {}\n'.format(bind_subs))

    print('Algorithmic Flops by hidden dimension, params, and per-batch-sample:')
    resolved_flops = alg_flops.subs(bind_subs)
    for enc_dim in encoder_dims:
        graph_params = resolved_params.subs({enc_hidden_dim_symbol: enc_dim})
        graph_flops = resolved_flops.subs({enc_hidden_dim_symbol: enc_dim})
        graph_flops_per_sample = float(graph_flops) / \
                                 bind_subs[subbatch_size_symbol]
        print('{}\t{}\t{}\t{}'.format(enc_dim, graph_params, graph_flops,
                                      int(graph_flops_per_sample)))

    print('\nAlgorithmic bytes accessed by hidden dimension, params:')
    resolved_bytes = alg_bytes.subs(bind_subs)
    for enc_dim in encoder_dims:
        graph_params = resolved_params.subs({enc_hidden_dim_symbol: enc_dim})
        graph_bytes = resolved_bytes.subs({enc_hidden_dim_symbol: enc_dim})
        print('{}\t{}\t{}'.format(enc_dim, graph_params, graph_bytes))

    print('\nAlgorithmic memory footprint by hidden dimension, params:')
    resolved_footprint = alg_footprint.subs(bind_subs)
    for enc_dim in encoder_dims:
        graph_params = resolved_params.subs({enc_hidden_dim_symbol: enc_dim})
        graph_footprint = resolved_footprint.subs({enc_hidden_dim_symbol: enc_dim})
        print('{}\t{}\t{}'.format(enc_dim, graph_params, graph_footprint))

    print('\nAlgorithmic minimal memory footprint by hidden dimension, params:')
    full_subs = dict(bind_subs)
    for enc_dim in encoder_dims:
        graph_params = resolved_params.subs({enc_hidden_dim_symbol: enc_dim})
        full_subs[enc_hidden_dim_symbol] = enc_dim
        graph_min_foot = graph.calcMinimalFootprint(symbol_subs=full_subs)
        print('{}\t{}\t{}'.format(enc_dim, graph_params, graph_min_foot))


if __name__ == "__main__":
    run_tf_speech_attention()
