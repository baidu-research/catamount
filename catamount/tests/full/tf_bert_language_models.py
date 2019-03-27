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

    # ============ TO REMOVE INITIALIZATION OPS! =============
    # NOTE: This code is pretty general and is likely to be migrated into
    # Catamount code for removing TF-specific initialization ops
    from catamount.ops import AssignOp
    from catamount.ops import VariableOp
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
    attn_head_size_symbol = utils.getIntSymbolFromString('attn_head_size')
    attn_heads_symbol = utils.getIntSymbolFromString('attn_heads')
    hidden_dim_symbol = attn_head_size_symbol * attn_heads_symbol
    vocab_size_symbol = utils.getIntSymbolFromString('vocab_size')
    sequence_length_symbol = utils.getIntSymbolFromString('seq_length')
    intermediate_dim_symbol = utils.getIntSymbolFromString('inter_dim')
    max_position_width_symbol = utils.getIntSymbolFromString('max_pos_width')
    graph_iters_symbol = utils.getIntSymbolFromString('graph::iters')
    # TODO(Joel): WHERE IS SUBBATCH SIZE?!

    # Convert these constant dimensions to symbols
    # TODO(Joel)
    # base_subbatch_size = None
    const_dict = {
                  'bert/embeddings/Reshape/shape': [1, sequence_length_symbol, hidden_dim_symbol],
                  'bert/embeddings/Reshape_2/shape': [1, sequence_length_symbol, hidden_dim_symbol],
                  'bert/embeddings/Reshape_3/shape': [1, sequence_length_symbol, hidden_dim_symbol],
                  'bert/embeddings/Slice/size': [sequence_length_symbol, -1],
                  'bert/encoder/Reshape/shape': [1, 1, sequence_length_symbol],
                  'bert/encoder/Reshape_1/shape': [-1, hidden_dim_symbol],
                  'bert/encoder/Reshape_[2-9]/shape': [1, sequence_length_symbol, hidden_dim_symbol],
                  'bert/encoder/Reshape_[1-9][0-9]/shape': [1, sequence_length_symbol, hidden_dim_symbol],
                  'bert/encoder/layer_[0-9]*/attention/self/Reshape/shape': [1, sequence_length_symbol, attn_heads_symbol, attn_head_size_symbol],
                  'bert/encoder/layer_[0-9]*/attention/self/Reshape_[1-2]/shape': [1, sequence_length_symbol, attn_heads_symbol, attn_head_size_symbol],
                  'bert/encoder/layer_[0-9]*/attention/self/Reshape_3/shape': [sequence_length_symbol, hidden_dim_symbol],
                 }
    graph.bindConstantValues(const_dict)


    # Next, bind the constant, placeholder, and variable shapes and propagate
    bind_dict = { # Constants
                  'bert/encoder/ones': [1, sequence_length_symbol, 1],
                  # Placeholders
                  'Placeholder': [1, sequence_length_symbol],
                  'Placeholder_1': [1, sequence_length_symbol],
                  'Placeholder_2': [1, sequence_length_symbol],
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
    assert graph.isValid()

    # TODO: base_subbatch_size = ???
    base_sequence_length = 128
    base_max_pos_width = 512
    if model_name == 'cased_L-12_H-768_A-12':
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
        # subbatch_size_symbol: base_subbatch_size,
        vocab_size_symbol: base_vocab_size,
        attn_heads_symbol: base_attn_heads,
        attn_head_size_symbol: base_attn_head_size,
        intermediate_dim_symbol: base_inter_dim,
        max_position_width_symbol: base_max_pos_width,
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
    pickle.dump(graph, open('catamount/frameworks/example_graphs/tensorflow/full_models/language_models/bert_models/graph_{}.p'.format(model_name), 'wb'))

    if is_pytest_run:
        return

    # TODO(Joel): For more advanced testing, continue from here
    return

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


if __name__ == "__main__":
    model_choices = ['cased_L-12_H-768_A-12',
                     'cased_L-24_H-1024_A-16',
                     'chinese_L-12_H-768_A-12',
                     'multi_cased_L-12_H-768_A-12',
                     'multilingual_L-12_H-768_A-12',
                     'uncased_L-12_H-768_A-12',
                     'uncased_L-24_H-1024_A-16',
                    ]
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', choices=model_choices, required=True,
                        help='The model to test ({})'.format(model_choices))
    args = parser.parse_args()

    run_tf_bert_lm(model_name=args.model_name)
