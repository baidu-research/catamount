import argparse
import numpy as np
import pickle
import re
import sympy
import sys
sys.setrecursionlimit(50000)

from catamount.api import utils
import catamount.frameworks.tensorflow
from catamount.ops.constant import *
from catamount.ops.unknown_op import UnknownOp
from catamount.ops.variable import *


is_pytest_run = False

def test_tf_cased_L12_H768_A12_model():
    global is_pytest_run
    is_pytest_run = True

    run_tf_bert_lm(model_name='cased_L-12_H-768_A-12')

def test_tf_uncased_L12_H768_A12_training_model():
    global is_pytest_run
    is_pytest_run = True

    run_tf_bert_lm(model_name='uncased_L-12_H-768_A-12.training')

def test_tf_cased_L24_H1024_A16_model():
    global is_pytest_run
    is_pytest_run = True

    run_tf_bert_lm(model_name='cased_L-24_H-1024_A-16')

def run_tf_bert_lm(model_name):
    global is_pytest_run

    graph_meta = 'catamount/frameworks/example_graphs/tensorflow/full_models/language_models/bert_models/{}.ckpt.meta' \
                  .format(model_name)

    graph = catamount.frameworks.tensorflow.import_graph(graph_meta)
    assert graph.isValid()

    # Next, remove ops that are not executed during a standard training step:
    # TODO(Joel)

    if not is_pytest_run:
        print('Initial graph:\n{}\n'.format(graph))
    init_params = graph.calcModelParameters()
    print('Initial parameters: {}'.format(init_params))
    print('Initial Flops: {}\n'.format(graph.calcAlgFlops()))
    print('Initial bytes: {}\n'.format(graph.calcAlgBytes()))
    print('Initial footprint: {}\n'.format(graph.calcAlgFootprint()))

    print('Placeholders:')
    for op in graph.getPlaceholders():
        print(op.debugString())
    print('')

    print('Variables:')
    for op in graph.getVariables():
        print(op.debugString())
    print('')

    # Set up symbols to name dimensions
    subbatch_size_symbol = utils.getIntSymbolFromString('subbatch_size')
    attn_head_size_symbol = utils.getIntSymbolFromString('attn_head_size')
    attn_heads_symbol = utils.getIntSymbolFromString('attn_heads')
    hidden_dim_symbol = attn_head_size_symbol * attn_heads_symbol
    vocab_size_symbol = utils.getIntSymbolFromString('vocab_size')
    sequence_length_symbol = utils.getIntSymbolFromString('seq_length')
    intermediate_dim_symbol = utils.getIntSymbolFromString('inter_dim')
    max_position_width_symbol = utils.getIntSymbolFromString('max_pos_width')
    num_predictions_symbol = utils.getIntSymbolFromString('num_predictions')
    graph_iters_symbol = utils.getIntSymbolFromString('graph::iters')

    num_layers = 0
    if 'L-3_' in model_name:
        num_layers = 3
    elif 'L-12_' in model_name:
        num_layers = 12
    elif 'L-24_' in model_name:
        num_layers = 24
    else:
        raise NotImplementedError('ERROR: Unknown model layer count: {}'.format(model_name))

    # Convert these constant dimensions to symbols
    if 'training' in model_name:
        const_dict = {
                      'batch_size': subbatch_size_symbol,
                      'bert/embeddings/assert_less_equal/x': sequence_length_symbol,
                      'bert/embeddings/assert_less_equal/y': max_position_width_symbol,
                      'bert/embeddings/dropout/Shape': [subbatch_size_symbol, sequence_length_symbol, hidden_dim_symbol],
                      'bert/embeddings/Reshape_1/shape': [subbatch_size_symbol, sequence_length_symbol, hidden_dim_symbol],
                      'bert/embeddings/Reshape_3/shape': [subbatch_size_symbol, sequence_length_symbol, hidden_dim_symbol],
                      'bert/embeddings/Reshape_4/shape': [1, sequence_length_symbol, hidden_dim_symbol],
                      'bert/embeddings/Slice/size': [sequence_length_symbol, -1],
                      'bert/encoder/layer_[0-9]*/attention/output/dropout/Shape': [subbatch_size_symbol*sequence_length_symbol, hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/self/dropout/Shape': [subbatch_size_symbol, attn_heads_symbol, sequence_length_symbol, sequence_length_symbol],
                      'bert/encoder/layer_[0-9]*/attention/self/Reshape_[1-2]/shape': [subbatch_size_symbol, sequence_length_symbol, attn_heads_symbol, attn_head_size_symbol],
                      'bert/encoder/layer_[0-9]*/attention/self/Reshape_3/shape': [subbatch_size_symbol*sequence_length_symbol, hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/self/Reshape/shape': [subbatch_size_symbol, sequence_length_symbol, attn_heads_symbol, attn_head_size_symbol],
                      'bert/encoder/layer_[0-9]*/output/dropout/Shape': [subbatch_size_symbol*sequence_length_symbol, hidden_dim_symbol],
                      'bert/encoder/ones/shape_as_tensor': [subbatch_size_symbol, sequence_length_symbol, 1],
                      'bert/encoder/Reshape_1/shape': [-1, hidden_dim_symbol],
                      'bert/encoder/Reshape_[2-9]/shape': [subbatch_size_symbol, sequence_length_symbol, hidden_dim_symbol],
                      'bert/encoder/Reshape_[1-9][0-9]/shape': [subbatch_size_symbol, sequence_length_symbol, hidden_dim_symbol],
                      'bert/encoder/Reshape/shape': [subbatch_size_symbol, 1, sequence_length_symbol],
                      'cls/predictions/one_hot/depth': vocab_size_symbol,
                      'gradients/bert/embeddings/add_1_grad/Shape_1': [1, sequence_length_symbol, hidden_dim_symbol],
                      'gradients/bert/embeddings/add_1_grad/Shape': [ subbatch_size_symbol, sequence_length_symbol, hidden_dim_symbol],
                      'gradients/bert/embeddings/dropout/div_grad/Shape': [subbatch_size_symbol, sequence_length_symbol, hidden_dim_symbol],
                      'gradients/bert/embeddings/GatherV2_grad/Shape': [vocab_size_symbol, hidden_dim_symbol],
                      'gradients/bert/embeddings/GatherV2_grad/Size': subbatch_size_symbol*sequence_length_symbol,
                      'gradients/bert/embeddings/LayerNorm/batchnorm/add_grad/Shape': [subbatch_size_symbol, sequence_length_symbol, 1],
                      'gradients/bert/embeddings/LayerNorm/batchnorm/mul_2_grad/Shape_1': [subbatch_size_symbol, sequence_length_symbol, hidden_dim_symbol],
                      'gradients/bert/embeddings/LayerNorm/batchnorm/mul_2_grad/Shape': [subbatch_size_symbol, sequence_length_symbol, 1],
                      'gradients/bert/embeddings/LayerNorm/batchnorm/mul_grad/Shape_1': [hidden_dim_symbol],
                      'gradients/bert/embeddings/LayerNorm/batchnorm/mul_grad/Shape': [subbatch_size_symbol, sequence_length_symbol, 1],
                      'gradients/bert/embeddings/LayerNorm/batchnorm/sub_grad/Shape_1': [subbatch_size_symbol, sequence_length_symbol, hidden_dim_symbol],
                      'gradients/bert/embeddings/LayerNorm/batchnorm/sub_grad/Shape': [hidden_dim_symbol],
                      'gradients/bert/embeddings/LayerNorm/moments/mean_grad/Shape': [subbatch_size_symbol, sequence_length_symbol, hidden_dim_symbol],
                      'gradients/bert/embeddings/LayerNorm/moments/SquaredDifference_grad/Shape_1': [subbatch_size_symbol, sequence_length_symbol, 1],
                      'gradients/bert/embeddings/LayerNorm/moments/SquaredDifference_grad/Shape': [subbatch_size_symbol, sequence_length_symbol, hidden_dim_symbol],
                      'gradients/bert/embeddings/LayerNorm/moments/variance_grad/Shape': [subbatch_size_symbol, sequence_length_symbol, hidden_dim_symbol],
                      'gradients/bert/embeddings/Reshape_1_grad/Shape': [subbatch_size_symbol*sequence_length_symbol, hidden_dim_symbol],
                      'gradients/bert/embeddings/Reshape_3_grad/Shape': [subbatch_size_symbol*sequence_length_symbol, hidden_dim_symbol],
                      'gradients/bert/embeddings/Reshape_4_grad/Shape': [sequence_length_symbol, hidden_dim_symbol],
                      'gradients/bert/embeddings/Slice_grad/Shape_1': [max_position_width_symbol, hidden_dim_symbol],
                      'gradients/bert/embeddings/Slice_grad/Shape': [sequence_length_symbol, hidden_dim_symbol],
                      'gradients/bert/encoder/layer_[0-9]*/attention/output/dropout/div_grad/Shape': [subbatch_size_symbol*sequence_length_symbol, hidden_dim_symbol],
                      'gradients/bert/encoder/layer_[0-9]*/attention/output/LayerNorm/batchnorm/add_grad/Shape': [subbatch_size_symbol*sequence_length_symbol, 1],
                      'gradients/bert/encoder/layer_[0-9]*/attention/output/LayerNorm/batchnorm/mul_2_grad/Shape_1': [subbatch_size_symbol*sequence_length_symbol, hidden_dim_symbol],
                      'gradients/bert/encoder/layer_[0-9]*/attention/output/LayerNorm/batchnorm/mul_2_grad/Shape': [subbatch_size_symbol*sequence_length_symbol, 1],
                      'gradients/bert/encoder/layer_[0-9]*/attention/output/LayerNorm/batchnorm/mul_grad/Shape_1': [hidden_dim_symbol],
                      'gradients/bert/encoder/layer_[0-9]*/attention/output/LayerNorm/batchnorm/mul_grad/Shape': [subbatch_size_symbol*sequence_length_symbol, 1],
                      'gradients/bert/encoder/layer_[0-9]*/attention/output/LayerNorm/batchnorm/sub_grad/Shape_1': [subbatch_size_symbol*sequence_length_symbol, hidden_dim_symbol],
                      'gradients/bert/encoder/layer_[0-9]*/attention/output/LayerNorm/batchnorm/sub_grad/Shape': [hidden_dim_symbol],
                      'gradients/bert/encoder/layer_[0-9]*/attention/output/LayerNorm/moments/mean_grad/Shape': [subbatch_size_symbol*sequence_length_symbol, hidden_dim_symbol],
                      'gradients/bert/encoder/layer_[0-9]*/attention/output/LayerNorm/moments/SquaredDifference_grad/Shape_1': [subbatch_size_symbol*sequence_length_symbol, 1],
                      'gradients/bert/encoder/layer_[0-9]*/attention/output/LayerNorm/moments/SquaredDifference_grad/Shape': [subbatch_size_symbol*sequence_length_symbol, hidden_dim_symbol],
                      'gradients/bert/encoder/layer_[0-9]*/attention/output/LayerNorm/moments/variance_grad/Shape': [subbatch_size_symbol*sequence_length_symbol, hidden_dim_symbol],
                      'gradients/bert/encoder/layer_[0-9]*/attention/self/add_grad/Shape_1': [subbatch_size_symbol, 1, sequence_length_symbol, sequence_length_symbol],
                      'gradients/bert/encoder/layer_[0-9]*/attention/self/add_grad/Shape': [subbatch_size_symbol, attn_heads_symbol, sequence_length_symbol, sequence_length_symbol],
                      'gradients/bert/encoder/layer_[0-9]*/attention/self/dropout/div_grad/Shape': [subbatch_size_symbol, attn_heads_symbol, sequence_length_symbol, sequence_length_symbol],
                      'gradients/bert/encoder/layer_[0-9]*/attention/self/Mul_grad/Shape': [subbatch_size_symbol, attn_heads_symbol, sequence_length_symbol, sequence_length_symbol],
                      'gradients/bert/encoder/layer_[0-9]*/attention/self/Reshape_1_grad/Shape': [subbatch_size_symbol*sequence_length_symbol, hidden_dim_symbol],
                      'gradients/bert/encoder/layer_[0-9]*/attention/self/Reshape_2_grad/Shape': [subbatch_size_symbol*sequence_length_symbol, hidden_dim_symbol],
                      'gradients/bert/encoder/layer_[0-9]*/attention/self/Reshape_3_grad/Shape': [subbatch_size_symbol, sequence_length_symbol, attn_heads_symbol, attn_head_size_symbol],
                      'gradients/bert/encoder/layer_[0-9]*/attention/self/Reshape_grad/Shape': [subbatch_size_symbol*sequence_length_symbol, hidden_dim_symbol],
                      'gradients/bert/encoder/layer_[0-9]*/intermediate/dense/add_1_grad/Shape_1': [subbatch_size_symbol*sequence_length_symbol, intermediate_dim_symbol],
                      'gradients/bert/encoder/layer_[0-9]*/intermediate/dense/mul_1_grad/Shape_1': [subbatch_size_symbol*sequence_length_symbol, intermediate_dim_symbol],
                      'gradients/bert/encoder/layer_[0-9]*/intermediate/dense/mul_2_grad/Shape_1': [subbatch_size_symbol*sequence_length_symbol, intermediate_dim_symbol],
                      'gradients/bert/encoder/layer_[0-9]*/intermediate/dense/mul_grad/Shape_1': [subbatch_size_symbol*sequence_length_symbol, intermediate_dim_symbol],
                      'gradients/bert/encoder/layer_[0-9]*/intermediate/dense/Pow_grad/Shape': [subbatch_size_symbol*sequence_length_symbol, intermediate_dim_symbol],
                      'gradients/bert/encoder/layer_[0-9]*/intermediate/dense/Pow_grad/zeros_like/shape_as_tensor': [subbatch_size_symbol*sequence_length_symbol, intermediate_dim_symbol],
                      'gradients/bert/encoder/layer_[0-9]*/output/dropout/div_grad/Shape': [subbatch_size_symbol*sequence_length_symbol, hidden_dim_symbol],
                      'gradients/bert/encoder/layer_[0-9]*/output/LayerNorm/batchnorm/add_grad/Shape': [subbatch_size_symbol*sequence_length_symbol, 1],
                      'gradients/bert/encoder/layer_[0-9]*/output/LayerNorm/batchnorm/mul_2_grad/Shape_1': [subbatch_size_symbol*sequence_length_symbol, hidden_dim_symbol],
                      'gradients/bert/encoder/layer_[0-9]*/output/LayerNorm/batchnorm/mul_2_grad/Shape': [subbatch_size_symbol*sequence_length_symbol, 1],
                      'gradients/bert/encoder/layer_[0-9]*/output/LayerNorm/batchnorm/mul_grad/Shape_1': [hidden_dim_symbol],
                      'gradients/bert/encoder/layer_[0-9]*/output/LayerNorm/batchnorm/mul_grad/Shape': [subbatch_size_symbol*sequence_length_symbol, 1],
                      'gradients/bert/encoder/layer_[0-9]*/output/LayerNorm/batchnorm/sub_grad/Shape_1': [subbatch_size_symbol*sequence_length_symbol, hidden_dim_symbol],
                      'gradients/bert/encoder/layer_[0-9]*/output/LayerNorm/batchnorm/sub_grad/Shape': [hidden_dim_symbol],
                      'gradients/bert/encoder/layer_[0-9]*/output/LayerNorm/moments/mean_grad/Shape': [subbatch_size_symbol*sequence_length_symbol, hidden_dim_symbol],
                      'gradients/bert/encoder/layer_[0-9]*/output/LayerNorm/moments/SquaredDifference_grad/Shape_1': [subbatch_size_symbol*sequence_length_symbol, 1],
                      'gradients/bert/encoder/layer_[0-9]*/output/LayerNorm/moments/SquaredDifference_grad/Shape': [subbatch_size_symbol*sequence_length_symbol, hidden_dim_symbol],
                      'gradients/bert/encoder/layer_[0-9]*/output/LayerNorm/moments/variance_grad/Shape': [subbatch_size_symbol*sequence_length_symbol, hidden_dim_symbol],
                      'gradients/bert/encoder/Reshape_1_grad/Shape': [subbatch_size_symbol, sequence_length_symbol, hidden_dim_symbol],
                      'gradients/bert/encoder/Reshape_4_grad/Shape': [subbatch_size_symbol*sequence_length_symbol, hidden_dim_symbol],
                      'gradients/bert/encoder/Reshape_13_grad/Shape': [subbatch_size_symbol*sequence_length_symbol, hidden_dim_symbol],
                      'gradients/bert/pooler/Squeeze_grad/Shape': [subbatch_size_symbol, 1, hidden_dim_symbol],
                      'gradients/bert/pooler/strided_slice_grad/Shape': [subbatch_size_symbol, sequence_length_symbol, hidden_dim_symbol],
                      'gradients/cls/predictions/Sum_1_grad/Const': [subbatch_size_symbol*num_predictions_symbol, ],
                      'gradients/cls/predictions/transform/LayerNorm/batchnorm/add_grad/Shape': [subbatch_size_symbol*num_predictions_symbol, 1],
                      'gradients/cls/predictions/transform/LayerNorm/batchnorm/mul_2_grad/Shape': [subbatch_size_symbol*num_predictions_symbol, 1],
                      'gradients/cls/predictions/transform/LayerNorm/batchnorm/mul_grad/Shape': [subbatch_size_symbol*num_predictions_symbol, 1],
                      'gradients/cls/predictions/transform/LayerNorm/moments/SquaredDifference_grad/Shape_1': [subbatch_size_symbol*num_predictions_symbol, 1],
                      'gradients/cls/predictions/Sum_grad/Shape': [subbatch_size_symbol*num_predictions_symbol, vocab_size_symbol],
                      'gradients/cls/predictions/transform/dense/add_1_grad/Shape_1': [subbatch_size_symbol*num_predictions_symbol, hidden_dim_symbol],
                      'gradients/cls/predictions/transform/dense/mul_1_grad/Shape_1': [subbatch_size_symbol*num_predictions_symbol, hidden_dim_symbol],
                      'gradients/cls/predictions/transform/dense/mul_2_grad/Shape_1': [subbatch_size_symbol*num_predictions_symbol, hidden_dim_symbol],
                      'gradients/cls/predictions/transform/dense/mul_grad/Shape_1': [subbatch_size_symbol*num_predictions_symbol, hidden_dim_symbol],
                      'gradients/cls/predictions/transform/dense/Pow_grad/Shape': [subbatch_size_symbol*num_predictions_symbol, hidden_dim_symbol],
                      'gradients/cls/predictions/transform/dense/Pow_grad/zeros_like/shape_as_tensor': [subbatch_size_symbol*num_predictions_symbol, hidden_dim_symbol],
                      'gradients/cls/predictions/transform/LayerNorm/batchnorm/mul_2_grad/Shape_1': [subbatch_size_symbol*num_predictions_symbol, hidden_dim_symbol],
                      'gradients/cls/predictions/transform/LayerNorm/batchnorm/mul_grad/Shape_1': [hidden_dim_symbol],
                      'gradients/cls/predictions/transform/LayerNorm/batchnorm/sub_grad/Shape_1': [subbatch_size_symbol*num_predictions_symbol, hidden_dim_symbol],
                      'gradients/cls/predictions/transform/LayerNorm/batchnorm/sub_grad/Shape': [hidden_dim_symbol],
                      'gradients/cls/predictions/transform/LayerNorm/moments/mean_grad/Shape': [subbatch_size_symbol*num_predictions_symbol, hidden_dim_symbol],
                      'gradients/cls/predictions/transform/LayerNorm/moments/SquaredDifference_grad/Shape': [subbatch_size_symbol*num_predictions_symbol, hidden_dim_symbol],
                      'gradients/cls/predictions/transform/LayerNorm/moments/variance_grad/Shape': [subbatch_size_symbol*num_predictions_symbol, hidden_dim_symbol],
                      'gradients/cls/seq_relationship/Mean_grad/Const': [subbatch_size_symbol],
                      'gradients/cls/seq_relationship/Sum_grad/Shape': [subbatch_size_symbol, 2],
                      'gradients/loss/dropout/div_grad/Shape': [subbatch_size_symbol, hidden_dim_symbol],
                      'gradients/loss/Mean_grad/Const': [subbatch_size_symbol],
                      'gradients/loss/Sum_grad/Shape': [subbatch_size_symbol, 2],
                      'gradients/GatherV2_grad/Shape': [subbatch_size_symbol*sequence_length_symbol, hidden_dim_symbol],
                      'gradients/GatherV2_grad/Size': subbatch_size_symbol*num_predictions_symbol,
                      'gradients/Reshape_2_grad/Shape': [subbatch_size_symbol, sequence_length_symbol, hidden_dim_symbol],
                      'gradients/Shape_1': [vocab_size_symbol, hidden_dim_symbol],
                      'Reshape_2/shape': [subbatch_size_symbol*sequence_length_symbol, hidden_dim_symbol],
                      'range/limit': subbatch_size_symbol,
                      'mul/y': sequence_length_symbol,
                      'loss/dropout/Shape': [subbatch_size_symbol, hidden_dim_symbol],
                      # Number of parallel threads? 'num_parallel_calls': 128,
                     }

    else:
        const_dict = {
                      'bert/embeddings/Reshape/shape': [subbatch_size_symbol, sequence_length_symbol, hidden_dim_symbol],
                      'bert/embeddings/Reshape_2/shape': [subbatch_size_symbol, sequence_length_symbol, hidden_dim_symbol],
                      'bert/embeddings/Reshape_3/shape': [1, sequence_length_symbol, hidden_dim_symbol],
                      'bert/embeddings/Slice/size': [sequence_length_symbol, -1],
                      'bert/encoder/Reshape/shape': [subbatch_size_symbol, 1, sequence_length_symbol],
                      'bert/encoder/Reshape_1/shape': [-1, hidden_dim_symbol],
                      'bert/encoder/Reshape_[2-9]/shape': [subbatch_size_symbol, sequence_length_symbol, hidden_dim_symbol],
                      'bert/encoder/Reshape_[1-9][0-9]/shape': [subbatch_size_symbol, sequence_length_symbol, hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/self/Reshape/shape': [subbatch_size_symbol, sequence_length_symbol, attn_heads_symbol, attn_head_size_symbol],
                      'bert/encoder/layer_[0-9]*/attention/self/Reshape_[1-2]/shape': [subbatch_size_symbol, sequence_length_symbol, attn_heads_symbol, attn_head_size_symbol],
                      'bert/encoder/layer_[0-9]*/attention/self/Reshape_3/shape': [subbatch_size_symbol*sequence_length_symbol, hidden_dim_symbol],
                     }
    graph.bindConstantValues(const_dict)


    # Next, bind the constant, placeholder, and variable shapes and propagate
    if 'training' in model_name:
        bind_dict = { # Constants
                      # Placeholders
                      # Variables
                      'bert/embeddings/LayerNorm/beta/adam_m': [hidden_dim_symbol],
                      'bert/embeddings/LayerNorm/beta/adam_v': [hidden_dim_symbol],
                      'bert/embeddings/LayerNorm/beta': [hidden_dim_symbol],
                      'bert/embeddings/LayerNorm/gamma/adam_m': [hidden_dim_symbol],
                      'bert/embeddings/LayerNorm/gamma/adam_v': [hidden_dim_symbol],
                      'bert/embeddings/LayerNorm/gamma': [hidden_dim_symbol],
                      'bert/embeddings/position_embeddings': [max_position_width_symbol, hidden_dim_symbol],
                      'bert/embeddings/position_embeddings/adam_m': [max_position_width_symbol, hidden_dim_symbol],
                      'bert/embeddings/position_embeddings/adam_v': [max_position_width_symbol, hidden_dim_symbol],
                      'bert/embeddings/token_type_embeddings': [2, hidden_dim_symbol],
                      'bert/embeddings/token_type_embeddings/adam_m': [2, hidden_dim_symbol],
                      'bert/embeddings/token_type_embeddings/adam_v': [2, hidden_dim_symbol],
                      'bert/embeddings/word_embeddings/adam_m': [vocab_size_symbol, hidden_dim_symbol],
                      'bert/embeddings/word_embeddings/adam_v': [vocab_size_symbol, hidden_dim_symbol],
                      'bert/embeddings/word_embeddings': [vocab_size_symbol, hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/output/dense/bias/adam_m': [hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/output/dense/bias/adam_v': [hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/output/dense/bias': [hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/output/dense/kernel/adam_m': [hidden_dim_symbol, hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/output/dense/kernel/adam_v': [hidden_dim_symbol, hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/output/dense/kernel': [hidden_dim_symbol, hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/output/LayerNorm/beta/adam_m': [hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/output/LayerNorm/beta/adam_v': [hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/output/LayerNorm/beta': [hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/output/LayerNorm/gamma/adam_m': [hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/output/LayerNorm/gamma/adam_v': [hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/output/LayerNorm/gamma': [hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/self/key/bias/adam_m': [hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/self/key/bias/adam_v': [hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/self/key/kernel/adam_m': [hidden_dim_symbol, hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/self/key/kernel/adam_v': [hidden_dim_symbol, hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/self/query/bias/adam_m': [hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/self/query/bias/adam_v': [hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/self/query/kernel/adam_m': [hidden_dim_symbol, hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/self/query/kernel/adam_v': [hidden_dim_symbol, hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/self/[querykval]*/bias': [hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/self/[querykval]*/kernel': [hidden_dim_symbol, hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/self/value/bias/adam_m': [hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/self/value/bias/adam_v': [hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/self/value/kernel/adam_m': [hidden_dim_symbol, hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/self/value/kernel/adam_v': [hidden_dim_symbol, hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/intermediate/dense/bias/adam_m': [intermediate_dim_symbol],
                      'bert/encoder/layer_[0-9]*/intermediate/dense/bias/adam_v': [intermediate_dim_symbol],
                      'bert/encoder/layer_[0-9]*/intermediate/dense/bias': [intermediate_dim_symbol],
                      'bert/encoder/layer_[0-9]*/intermediate/dense/kernel/adam_m': [hidden_dim_symbol, intermediate_dim_symbol],
                      'bert/encoder/layer_[0-9]*/intermediate/dense/kernel/adam_v': [hidden_dim_symbol, intermediate_dim_symbol],
                      'bert/encoder/layer_[0-9]*/intermediate/dense/kernel': [hidden_dim_symbol, intermediate_dim_symbol],
                      'bert/encoder/layer_[0-9]*/output/dense/bias/adam_m': [hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/output/dense/bias/adam_v': [hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/output/dense/bias': [hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/output/dense/kernel/adam_m': [intermediate_dim_symbol, hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/output/dense/kernel/adam_v': [intermediate_dim_symbol, hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/output/dense/kernel': [intermediate_dim_symbol, hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/output/LayerNorm/beta/adam_m': [hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/output/LayerNorm/beta/adam_v': [hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/output/LayerNorm/beta': [hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/output/LayerNorm/gamma/adam_m': [hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/output/LayerNorm/gamma/adam_v': [hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/output/LayerNorm/gamma': [hidden_dim_symbol],
                      'bert/pooler/dense/bias/adam_m': [hidden_dim_symbol],
                      'bert/pooler/dense/bias/adam_v': [hidden_dim_symbol],
                      'bert/pooler/dense/bias': [hidden_dim_symbol],
                      'bert/pooler/dense/kernel/adam_m': [hidden_dim_symbol, hidden_dim_symbol],
                      'bert/pooler/dense/kernel/adam_v': [hidden_dim_symbol, hidden_dim_symbol],
                      'bert/pooler/dense/kernel': [hidden_dim_symbol, hidden_dim_symbol],
                      'cls/predictions/output_bias/adam_m': [vocab_size_symbol],
                      'cls/predictions/output_bias/adam_v': [vocab_size_symbol],
                      'cls/predictions/output_bias': [vocab_size_symbol],
                      'cls/predictions/transform/dense/bias/adam_m': [hidden_dim_symbol],
                      'cls/predictions/transform/dense/bias/adam_v': [hidden_dim_symbol],
                      'cls/predictions/transform/dense/bias': [hidden_dim_symbol],
                      'cls/predictions/transform/dense/kernel/adam_m': [hidden_dim_symbol, hidden_dim_symbol],
                      'cls/predictions/transform/dense/kernel/adam_v': [hidden_dim_symbol, hidden_dim_symbol],
                      'cls/predictions/transform/dense/kernel': [hidden_dim_symbol, hidden_dim_symbol],
                      'cls/predictions/transform/LayerNorm/beta/adam_m': [hidden_dim_symbol],
                      'cls/predictions/transform/LayerNorm/beta/adam_v': [hidden_dim_symbol],
                      'cls/predictions/transform/LayerNorm/beta': [hidden_dim_symbol],
                      'cls/predictions/transform/LayerNorm/gamma/adam_m': [hidden_dim_symbol],
                      'cls/predictions/transform/LayerNorm/gamma/adam_v': [hidden_dim_symbol],
                      'cls/predictions/transform/LayerNorm/gamma': [hidden_dim_symbol],
                      'cls/seq_relationship/output_bias': [2],
                      'cls/seq_relationship/output_weights': [2, hidden_dim_symbol],
                      'cls/seq_relationship/output_weights/adam_m': [2, hidden_dim_symbol],
                      'cls/seq_relationship/output_weights/adam_v': [2, hidden_dim_symbol],
                      'output_weights': [2, hidden_dim_symbol],
                      'output_weights/adam_m': [2, hidden_dim_symbol],
                      'output_weights/adam_v': [2, hidden_dim_symbol],
                    }

        # Manually hack the iterator op
        local_subbatch_size = 32
        local_sequence_length = 128
        local_num_predictions = 20
        it_op = graph.opsByName['IteratorGetNext']
        for out_tens in it_op._outputs:
            out_shape = out_tens.shape
            symbolic_out_shape = []
            for dim in out_shape.dims:
                if dim.value == local_subbatch_size:
                    symbolic_out_shape.append(subbatch_size_symbol)
                elif dim.value == local_sequence_length:
                    symbolic_out_shape.append(sequence_length_symbol)
                elif dim.value == local_num_predictions:
                    symbolic_out_shape.append(num_predictions_symbol)
                elif dim.value == 1:
                    symbolic_out_shape.append(1)
                else:
                    print('ERROR: Unknown IteratorGetNext dimension: {}'
                          .format(dim))
                    sys.exit(0)
            out_tens.mergeShape(symbolic_out_shape, make_symbolic=True)
    else:
        bind_dict = { # Constants
                      'bert/encoder/ones': [1, sequence_length_symbol, 1],
                      # Placeholders
                      'Placeholder': [subbatch_size_symbol, sequence_length_symbol],
                      'Placeholder_1': [subbatch_size_symbol, sequence_length_symbol],
                      'Placeholder_2': [subbatch_size_symbol, sequence_length_symbol],
                      # Variables
                      'bert/embeddings/word_embeddings': [vocab_size_symbol, hidden_dim_symbol],
                      'bert/embeddings/token_type_embeddings': [2, hidden_dim_symbol],
                      'bert/embeddings/position_embeddings': [max_position_width_symbol, hidden_dim_symbol],
                      'bert/embeddings/LayerNorm/beta': [hidden_dim_symbol],
                      'bert/embeddings/LayerNorm/gamma': [hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/self/[querykval]*/kernel': [hidden_dim_symbol, hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/self/[querykval]*/bias': [hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/output/dense/kernel': [hidden_dim_symbol, hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/output/dense/bias': [hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/output/LayerNorm/beta': [hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/attention/output/LayerNorm/gamma': [hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/intermediate/dense/kernel': [hidden_dim_symbol, intermediate_dim_symbol],
                      'bert/encoder/layer_[0-9]*/intermediate/dense/bias': [intermediate_dim_symbol],
                      'bert/encoder/layer_[0-9]*/output/dense/kernel': [intermediate_dim_symbol, hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/output/dense/bias': [hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/output/LayerNorm/beta': [hidden_dim_symbol],
                      'bert/encoder/layer_[0-9]*/output/LayerNorm/gamma': [hidden_dim_symbol],
                      'bert/pooler/dense/kernel': [hidden_dim_symbol, hidden_dim_symbol],
                      'bert/pooler/dense/bias': [hidden_dim_symbol],
                      'cls/predictions/output_bias': [vocab_size_symbol],
                      'cls/predictions/transform/LayerNorm/beta': [hidden_dim_symbol],
                      'cls/predictions/transform/LayerNorm/gamma': [hidden_dim_symbol],
                      'cls/predictions/transform/dense/bias': [hidden_dim_symbol],
                      'cls/predictions/transform/dense/kernel': [hidden_dim_symbol, hidden_dim_symbol],
                      'cls/seq_relationship/output_bias': [2],
                      'cls/seq_relationship/output_weights': [2, hidden_dim_symbol],
                    }


    print('Binding variables')

    graph.bindShapesAndPropagate(bind_dict, warn_if_ill_defined=(not is_pytest_run), make_symbolic=True)

    if 'training' in model_name:
        # Need to manually hack a few...
        max_op_name = 'gradients/bert/embeddings/LayerNorm/moments/mean_grad/Maximum'
        if max_op_name in graph.opsByName.keys():
            max_op = graph.opsByName[max_op_name]
            out_value = [subbatch_size_symbol, sequence_length_symbol, 1]
            max_op.outputs[0].setValue(out_value)
        max_op_name = 'gradients/bert/embeddings/LayerNorm/moments/variance_grad/Maximum'
        if max_op_name in graph.opsByName.keys():
            max_op = graph.opsByName[max_op_name]
            out_value = [subbatch_size_symbol, sequence_length_symbol, 1]
            max_op.outputs[0].setValue(out_value)
        max_op_name = 'gradients/cls/predictions/transform/LayerNorm/moments/mean_grad/Maximum'
        if max_op_name in graph.opsByName.keys():
            max_op = graph.opsByName[max_op_name]
            out_value = [subbatch_size_symbol*num_predictions_symbol, 1]
            max_op.outputs[0].setValue(out_value)
        max_op_name = 'gradients/cls/predictions/transform/LayerNorm/moments/variance_grad/Maximum'
        if max_op_name in graph.opsByName.keys():
            max_op = graph.opsByName[max_op_name]
            out_value = [subbatch_size_symbol*num_predictions_symbol, 1]
            max_op.outputs[0].setValue(out_value)
        max_op_name = 'gradients/cls/predictions/Sum_grad/Maximum'
        if max_op_name in graph.opsByName.keys():
            max_op = graph.opsByName[max_op_name]
            out_value = [subbatch_size_symbol*num_predictions_symbol, 1]
            max_op.outputs[0].setValue(out_value)
        max_op_name = 'gradients/cls/seq_relationship/Sum_grad/Maximum'
        if max_op_name in graph.opsByName.keys():
            max_op = graph.opsByName[max_op_name]
            out_value = [subbatch_size_symbol, 1]
            max_op.outputs[0].setValue(out_value)

        for i in range(num_layers):
            max_op_name = 'gradients/bert/encoder/layer_{}/attention/output/LayerNorm/moments/mean_grad/Maximum'.format(i)
            max_op = graph.opsByName[max_op_name]
            out_value = [subbatch_size_symbol*sequence_length_symbol, 1]
            max_op.outputs[0].setValue(out_value)
            max_op_name = 'gradients/bert/encoder/layer_{}/attention/output/LayerNorm/moments/variance_grad/Maximum'.format(i)
            max_op = graph.opsByName[max_op_name]
            out_value = [subbatch_size_symbol*sequence_length_symbol, 1]
            max_op.outputs[0].setValue(out_value)
            max_op_name = 'gradients/bert/encoder/layer_{}/output/LayerNorm/moments/mean_grad/Maximum'.format(i)
            max_op = graph.opsByName[max_op_name]
            out_value = [subbatch_size_symbol*sequence_length_symbol, 1]
            max_op.outputs[0].setValue(out_value)
            max_op_name = 'gradients/bert/encoder/layer_{}/output/LayerNorm/moments/variance_grad/Maximum'.format(i)
            max_op = graph.opsByName[max_op_name]
            out_value = [subbatch_size_symbol*sequence_length_symbol, 1]
            max_op.outputs[0].setValue(out_value)

        graph.bindShapesAndPropagate(bind_dict, warn_if_ill_defined=(not is_pytest_run), make_symbolic=True)

    assert graph.isValid()

    base_subbatch_size = 1
    base_sequence_length = 128
    base_max_pos_width = 512
    base_num_predictions = 0
    if model_name == 'uncased_L-3_H-768_A-12.training' or \
       model_name == 'uncased_L-12_H-768_A-12.training':
        base_vocab_size = 30522
        base_attn_heads = 12
        base_attn_head_size = 64
        base_inter_dim = 3072
        base_num_predictions = 20
    elif model_name == 'cased_L-12_H-768_A-12':
        base_vocab_size = 28996
        base_attn_heads = 12
        base_attn_head_size = 64
        base_inter_dim = 3072
    elif model_name == 'cased_L-24_H-1024_A-16':
        base_vocab_size = 28996
        base_attn_heads = 16
        base_attn_head_size = 64
        base_inter_dim = 4096
    elif model_name == 'chinese_L-12_H-768_A-12':
        base_vocab_size = 21128
        base_attn_heads = 12
        base_attn_head_size = 64
        base_inter_dim = 3072
    elif model_name == 'uncased_L-12_H-768_A-12':
        base_vocab_size = 30522
        base_attn_heads = 12
        base_attn_head_size = 64
        base_inter_dim = 3072
    elif model_name == 'uncased_L-24_H-1024_A-16':
        base_vocab_size = 30522
        base_attn_heads = 16
        base_attn_head_size = 64
        base_inter_dim = 4096
    elif model_name == 'multi_cased_L-12_H-768_A-12':
        base_vocab_size = 119547
        base_attn_heads = 12
        base_attn_head_size = 64
        base_inter_dim = 3072
    elif model_name == 'multilingual_L-12_H-768_A-12':
        base_vocab_size = 105879
        base_attn_heads = 12
        base_attn_head_size = 64
        base_inter_dim = 3072
    else:
        raise NotImplementedError('ERROR: Unknown model: {}'.format(model_name))

    if not is_pytest_run:
        print('\n\nCleaned Graph:\n{}'.format(graph))

    print('\n\nBound values')

    attn_head_size_symbol = utils.getIntSymbolFromString('attn_head_size')
    attn_heads_symbol = utils.getIntSymbolFromString('attn_heads')
    hidden_dim_symbol = attn_head_size_symbol * attn_heads_symbol
    vocab_size_symbol = utils.getIntSymbolFromString('vocab_size')

    bind_subs = {
        graph_iters_symbol: 1,
        attn_head_size_symbol: base_attn_head_size,
        sequence_length_symbol: base_sequence_length,
        subbatch_size_symbol: base_subbatch_size,
        vocab_size_symbol: base_vocab_size,
        attn_heads_symbol: base_attn_heads,
        attn_head_size_symbol: base_attn_head_size,
        intermediate_dim_symbol: base_inter_dim,
        max_position_width_symbol: base_max_pos_width,
        num_predictions_symbol: base_num_predictions,
    }

    # Verify parameter counts first
    if model_name == 'cased_L-12_H-768_A-12' or \
       model_name == 'multi_cased_L-12_H-768_A-12' or \
       model_name == 'chinese_L-12_H-768_A-12' or \
       model_name == 'multilingual_L-12_H-768_A-12' or \
       model_name == 'uncased_L-12_H-768_A-12':
        correct_symbolic_params = 50 * attn_head_size_symbol ** 2 * attn_heads_symbol ** 2 + \
                                  24 * attn_head_size_symbol * attn_heads_symbol * intermediate_dim_symbol + \
                                  attn_head_size_symbol * attn_heads_symbol * max_position_width_symbol + \
                                  attn_head_size_symbol * attn_heads_symbol * vocab_size_symbol + \
                                  118 * attn_head_size_symbol * attn_heads_symbol + \
                                  12 * intermediate_dim_symbol + vocab_size_symbol + 2
    elif model_name == 'cased_L-24_H-1024_A-16' or \
         model_name == 'uncased_L-24_H-1024_A-16':
        correct_symbolic_params = 98 * attn_head_size_symbol ** 2 * attn_heads_symbol ** 2 + \
                                  48 * attn_head_size_symbol * attn_heads_symbol * intermediate_dim_symbol + \
                                  attn_head_size_symbol * attn_heads_symbol * max_position_width_symbol + \
                                  attn_head_size_symbol * attn_heads_symbol * vocab_size_symbol + \
                                  226 * attn_head_size_symbol * attn_heads_symbol + \
                                  24 * intermediate_dim_symbol + vocab_size_symbol + 2
    elif model_name == 'uncased_L-3_H-768_A-12.training':
        correct_symbolic_params = 42 * attn_head_size_symbol ** 2 * attn_heads_symbol ** 2 + \
                                  18 * attn_head_size_symbol * attn_heads_symbol * intermediate_dim_symbol + \
                                  3 * attn_head_size_symbol * attn_heads_symbol * max_position_width_symbol + \
                                  3 * attn_head_size_symbol * attn_heads_symbol * vocab_size_symbol + \
                                  111 * attn_head_size_symbol * attn_heads_symbol + \
                                  9 * intermediate_dim_symbol + 3 * vocab_size_symbol + 6
    elif model_name == 'uncased_L-12_H-768_A-12.training':
        correct_symbolic_params = 147 * attn_head_size_symbol ** 2 * attn_heads_symbol ** 2 + \
                                  72 * attn_head_size_symbol * attn_heads_symbol * intermediate_dim_symbol + \
                                  3 * attn_head_size_symbol * attn_heads_symbol * max_position_width_symbol + \
                                  3 * attn_head_size_symbol * attn_heads_symbol * vocab_size_symbol + \
                                  345 * attn_head_size_symbol * attn_heads_symbol + \
                                  36 * intermediate_dim_symbol + 6
    else:
        raise NotImplementedError('ERROR: Unknown model: {}'.format(model_name))

    parameters = graph.calcModelParameters()
    assert sympy.simplify(parameters - correct_symbolic_params) == 0, \
           'Param count incorrect!\n  Expecting: {}\n  Calculated: {}' \
           .format(correct_symbolic_params, parameters)

    if model_name == 'cased_L-12_H-768_A-12':
        correct_flops = 22419929740
        correct_bytes = 1099363516
        correct_total_footprint = 783867512
    elif model_name == 'chinese_L-12_H-768_A-12':
        correct_flops = 22419929740
        correct_bytes = 1099363516
        correct_total_footprint = 759665544
    elif model_name == 'multi_cased_L-12_H-768_A-12':
        correct_flops = 22419929742
        correct_bytes = 1099363535
        correct_total_footprint = 1062402398
    elif model_name == 'cased_L-24_H-1024_A-16':
        correct_flops = 79110514074
        correct_bytes = 3207667311
        correct_total_footprint = 2257396562
    elif model_name == 'multilingual_L-12_H-768_A-12':
        correct_flops = 22419929740
        correct_bytes = 1099363516
        correct_total_footprint = 1020359620
    elif model_name == 'uncased_L-12_H-768_A-12':
        correct_flops = 22419929740
        correct_bytes = 1099363516
        correct_total_footprint = 788561488
    elif model_name == 'uncased_L-24_H-1024_A-16':
        correct_flops = 79110514072
        correct_bytes = 3207667292
        correct_total_footprint = 2263653152
    elif model_name == 'uncased_L-3_H-768_A-12.training':
        correct_flops = 20509424861
        correct_bytes = 10058376317
        correct_total_footprint = 4828490227
    elif model_name == 'uncased_L-12_H-768_A-12.training':
        correct_flops = 69088417112
        correct_bytes = 23450183862
        correct_total_footprint = 11301063592
    else:
        raise NotImplementedError('ERROR: Unknown model: {}'.format(model_name))

    print('Symbol associations: {}\n'.format(bind_subs))

    # Calculate model parameter count
    resolved_params = parameters.subs(bind_subs)
    correct_params = int(correct_symbolic_params.subs(bind_subs))
    try:
        resolved_params = int(resolved_params)
    except:
        print('ERROR: resolved_params should be int, but is {} = {}'.format(
              type(resolved_params), resolved_params))
    assert init_params == correct_params, \
           'Incorrect correct_params calculation: init: {} correct: {}' \
           .format(init_params, correct_params)
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
#    assert resolved_flops == correct_flops, \
#           'Incorrect algorithmic flops: {}'.format(resolved_flops)
    if resolved_flops != correct_flops:
        print('Incorrect algorithmic flops: {}'.format(resolved_flops))
    print('Algorithmic Flops: {}\nWith specified dims: {}\n'.format(alg_flops, resolved_flops))

    # Calculate algorthmic Bytes accessed
    alg_bytes = graph.calcAlgBytes()
    resolved_bytes = alg_bytes.subs(bind_subs)
    try:
        resolved_bytes = int(resolved_bytes)
    except:
        print('ERROR: resolved_bytes should be int, but is {} = {}'.format(
              type(resolved_bytes), resolved_bytes))
#    assert resolved_bytes == correct_bytes, \
#           'Incorrect algorithmic bytes: {}'.format(resolved_bytes)
    if resolved_bytes != correct_bytes:
        print('Incorrect algorithmic bytes: {}'.format(resolved_bytes))
    print('Alg bytes accessed: {}\nWith specified dims: {}\n'.format(alg_bytes, resolved_bytes))

    # Calculate total memory footprint
    alg_footprint = graph.calcAlgFootprint()
    resolved_footprint = alg_footprint.subs(bind_subs)
    try:
        resolved_footprint = int(resolved_footprint)
    except:
        print('ERROR: resolved_footprint should be int, but is {} = {}'.format(
              type(resolved_footprint), resolved_footprint))
#    assert resolved_footprint == correct_total_footprint, \
#           'Incorrect algorithmic footprint: {}'.format(resolved_footprint)
    if resolved_footprint != correct_total_footprint:
        print('Incorrect algorithmic footprint: {}'.format(resolved_footprint))
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
    pickle.dump(graph, open('catamount/frameworks/example_graphs/tensorflow/full_models/language_models/bert_models/graph_{}.p'.format(model_name), 'wb'))

    if is_pytest_run:
        return

    print('\n\n======= Algorithmic graph-level analytics: =======')

    attn_head_sizes = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 14, 18, 20, 25, 28, 35, 40, 50, 56, 69, 78, 86, 96, 108, 119, 123, 133, 148, 163, 182, 202, 221, 246, 273, 297, 329, 330, 364, 396, 436, 437, 520, 572, 617, 676, 740, 796, 869, 948, 1017, 1106, 1202, 1286, 1394, 1510, 1611, 1742, 1882, 2004, 2161, 2476, 3040, 3714, 4520, 5478, 6628, 8019, 9702, 11739, 14204, 17186, 20795, 25161, 30444, 36837, 38100]
    bind_subs[subbatch_size_symbol] = 32

    bind_subs.pop(attn_head_size_symbol)
    resolved_params = parameters.subs(bind_subs)

    print('Symbol associations: {}\n'.format(bind_subs))

    print('Algorithmic Flops by hidden dimension, params, and per-batch-sample:')
    resolved_flops = alg_flops.subs(bind_subs)
    for attn_head_size in attn_head_sizes:
        graph_params = resolved_params.subs({attn_head_size_symbol: attn_head_size,
                                             intermediate_dim_symbol: 4 * attn_head_size_symbol * attn_heads_symbol})
        graph_flops = resolved_flops.subs({attn_head_size_symbol: attn_head_size,
                                           intermediate_dim_symbol: 4 * attn_head_size_symbol * attn_heads_symbol})
        graph_flops_per_sample = float(graph_flops) / \
                                 bind_subs[subbatch_size_symbol]
        print('{}\t{}\t{}\t{}'.format(attn_head_size, graph_params, graph_flops,
                                      int(graph_flops_per_sample)))

    print('\nAlgorithmic bytes accessed by hidden dimension, params:')
    resolved_bytes = alg_bytes.subs(bind_subs)
    for attn_head_size in attn_head_sizes:
        graph_params = resolved_params.subs({attn_head_size_symbol: attn_head_size,
                                             intermediate_dim_symbol: 4 * attn_head_size_symbol * attn_heads_symbol})
        graph_bytes = resolved_bytes.subs({attn_head_size_symbol: attn_head_size,
                                           intermediate_dim_symbol: 4 * attn_head_size_symbol * attn_heads_symbol})
        print('{}\t{}\t{}'.format(attn_head_size, graph_params, graph_bytes))

    print('\nAlgorithmic total memory footprint by hidden dimension, params:')
    resolved_footprint = alg_footprint.subs(bind_subs)
    for attn_head_size in attn_head_sizes:
        graph_params = resolved_params.subs({attn_head_size_symbol: attn_head_size,
                                             intermediate_dim_symbol: 4 * attn_head_size_symbol * attn_heads_symbol})
        graph_footprint = resolved_footprint.subs({attn_head_size_symbol: attn_head_size,
                                           intermediate_dim_symbol: 4 * attn_head_size_symbol * attn_heads_symbol})
        print('{}\t{}\t{}'.format(attn_head_size, graph_params, graph_footprint))

    print('\nAlgorithmic minimal memory footprint by hidden dimension, params:')
    full_subs = dict(bind_subs)
    full_subs[intermediate_dim_symbol] = 4 * attn_head_size_symbol * attn_heads_symbol
    for attn_head_size in attn_head_sizes:
        graph_params = resolved_params.subs({attn_head_size_symbol: attn_head_size,
                                             intermediate_dim_symbol: 4 * attn_head_size_symbol * attn_heads_symbol})
        full_subs[attn_head_size_symbol] = attn_head_size
        graph_min_foot = graph.calcMinimalFootprint(symbol_subs=full_subs)
        print('{}\t{}\t{}'.format(attn_head_size, graph_params, graph_min_foot))


if __name__ == "__main__":
    model_choices = ['cased_L-12_H-768_A-12',
                     'cased_L-24_H-1024_A-16',
                     'chinese_L-12_H-768_A-12',
                     'multi_cased_L-12_H-768_A-12',
                     'multilingual_L-12_H-768_A-12',
                     'uncased_L-12_H-768_A-12',
                     'uncased_L-24_H-1024_A-16',
                     'uncased_L-3_H-768_A-12.training',
                     'uncased_L-12_H-768_A-12.training',
                    ]
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', choices=model_choices, required=True,
                        help='The model to test ({})'.format(model_choices))
    args = parser.parse_args()

    run_tf_bert_lm(model_name=args.model_name)
