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

def test_tf_w2v_model():
    global is_pytest_run
    is_pytest_run = True
    run_tf_w2v_model()

def run_tf_w2v_model():
    global is_pytest_run

    graph_meta = 'catamount/frameworks/example_graphs/tensorflow/full_models/language_models/word2vec_n200-latest_model.meta'

    graph = catamount.frameworks.tensorflow.import_graph(graph_meta)
    assert graph.isValid()

    # Next, remove ops that are not executed during a standard training step:
    graph_ops = list(graph._ops_by_name.values())
    for op in graph_ops:
         # Certain ops are only used for inference
         if 'Model/NceLoss_1_3/' in op.name or \
            'Model/Collapse_1/' in op.name or \
            'Model/Embedding_1_3/' in op.name or \
            'Model/Labels_1/' in op.name or \
            'Model/SkipGramSampler_1/' in op.name or \
            'Model/Mask_1/' in op.name:
             graph.removeOp(op)
         elif \
              op.name == 'Model/Cast_1' or \
              op.name == 'Model/Sum_1' or \
              op.name == 'Model/Size_1' or \
              op.name == 'Model/Exp_1' or \
              op.name == 'Model/truediv_2' or \
              op.name == 'Model/truediv_3':
             graph.removeOp(op)

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
    skip_window_symbol = utils.getPositiveIntSymbolFromString('skip_window')
    num_skips_symbol = utils.getPositiveIntSymbolFromString('num_skips')
    nce_samples_symbol = utils.getPositiveIntSymbolFromString('nce_samples')
    hidden_dim_symbol = utils.getIntSymbolFromString('hidden_dim')
    vocab_size_symbol = utils.getIntSymbolFromString('vocab_size')
    subbatch_size_symbol = utils.getIntSymbolFromString('subbatch_size')
    sequence_length_symbol = utils.getIntSymbolFromString('sequence_length')
    batch_times_seq_symbol = sequence_length_symbol * subbatch_size_symbol
    graph_iters_symbol = utils.getIntSymbolFromString('graph::iters')
    # For simplicity, assign samples symbol in the op
    nce_samp_op = graph.opsByName['Model/NceLoss_1_1/nce_loss/LogUniformCandidateSampler']
    nce_samp_op._num_samples_symbol = nce_samples_symbol

    # Convert these constant dimensions to symbols
    base_skip_window = 8
    base_num_skips = 8
    base_nce_samples = 64
    base_hidden_dim = 400
    base_vocab_size = 40004
    base_sequence_length = 32
    base_subbatch_size = 1

    # Find and set constants that contain model hyperparameters
    const_dict = { 'Model/Gradient/Compute/gradients/Model/NceLoss_1_1/nce_loss/sub_1_grad/Shape_1': [nce_samples_symbol],
                   'Model/SkipGramSampler/Const': 2 * skip_window_symbol,
                   'Model/SkipGramSampler/strided_slice/stack': [0, skip_window_symbol],
                   'Model/SkipGramSampler/strided_slice/stack_1': [0, -skip_window_symbol],
                   'Model/Collapse/Reshape/shape': [-1, hidden_dim_symbol],
                   'Model/Gradient/Compute/gradients/Model/Embedding_1_1/Gather_grad/Shape': [vocab_size_symbol, hidden_dim_symbol],
                   'Model/Gradient/Compute/gradients/Model/NceLoss_1_1/nce_loss/embedding_lookup_1_grad/Shape': [vocab_size_symbol],
                   'Model/Gradient/Compute/gradients/Model/NceLoss_1_1/nce_loss/embedding_lookup_grad/Shape': [vocab_size_symbol, hidden_dim_symbol],
                   'Model/Mask/NotEqual/y': vocab_size_symbol - 3,
                   'Model/SkipGramSampler/Const_2': num_skips_symbol,
                   'Model/SkipGramSampler/Tile_1/multiples': [1, num_skips_symbol],
                   'Model/SkipGramSampler/Tile/multiples': [1, num_skips_symbol],
                 }
    graph.bindConstantValues(const_dict)

    # Next, bind the constant, placeholder, and variable shapes and propagate
    bind_dict = { # Constants
                  # Placeholders
                  'Input/Input': [subbatch_size_symbol, sequence_length_symbol],
                  'Labels/Labels': [subbatch_size_symbol, sequence_length_symbol],
                  # Variables
                  'Model/NceLoss_1/b_Softmax': [vocab_size_symbol],
                  'Model/NceLoss_1/W_Softmax': [vocab_size_symbol, hidden_dim_symbol],
                  'Model/Embedding_1/EmbeddingWeights': [vocab_size_symbol, hidden_dim_symbol],
                }

    print('Binding variables')

    # HACK: For now, manually set GatherNd op shapes. Later, implement GatherNd
    gnd_op = graph.opsByName['Model/SkipGramSampler/GatherNd']
    gnd_op.outputs[0].mergeShape([subbatch_size_symbol, num_skips_symbol * (sequence_length_symbol - 2 * skip_window_symbol)])

    graph.bindShapesAndPropagate(bind_dict, warn_if_ill_defined=(not is_pytest_run), make_symbolic=True)
    assert graph.isValid()

    if not is_pytest_run:
        print('\n\nCleaned Graph:\n{}'.format(graph))

    print('\n\nBound values')

    bind_subs = {
        graph_iters_symbol: 1,
        hidden_dim_symbol: base_hidden_dim,
        sequence_length_symbol: base_sequence_length,
        subbatch_size_symbol: base_subbatch_size,
        vocab_size_symbol: base_vocab_size,
        skip_window_symbol: base_skip_window,
        num_skips_symbol: base_num_skips,
        nce_samples_symbol: base_nce_samples,
    }

    # Verify parameter counts first
    parameters = graph.calcModelParameters()

    correct_params = 32043205
    correct_flops = 21148823
    correct_bytes = 23762537
    correct_total_footprint = 137949925

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
    pickle.dump(graph, open('catamount/frameworks/example_graphs/tensorflow/full_models/language_models/graph_word2vec.p', 'wb'))

    if is_pytest_run:
        return

    print('\n\n======= Algorithmic graph-level analytics: =======')

    hidden_dims = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 14, 18, 20, 25, 28, 35, 40, 50, 56, 69, 78, 86, 96, 108, 119, 123, 133, 148, 163, 182, 202, 221, 246, 273, 297, 329, 330, 364, 396, 436, 437, 520, 572, 617, 676, 740, 796, 869, 948, 1017, 1106, 1202, 1286, 1394, 1510, 1611, 1742, 1882, 2004, 2161, 2476, 3040, 3714, 4520, 5478, 6628, 8019, 9702, 11739, 14204, 17186, 20795, 25161, 30444, 36837, 38100]

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


if __name__ == "__main__":
    test_tf_w2v_model()



