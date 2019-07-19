import argparse
import numpy as np
import os
import pickle
import sympy
import sys
sys.setrecursionlimit(50000)

from catamount.api import utils
import catamount.frameworks.tensorflow
from catamount.ops.constant import *
from catamount.ops.variable import *
from catamount.ops.math_ops import MaximumOp


is_pytest_run = False

def test_tf_image_resnet_18():
    global is_pytest_run
    is_pytest_run = True

    run_tf_image_resnet(depth=18, filter_scale=1.0)

def test_tf_image_resnet_34():
    global is_pytest_run
    is_pytest_run = True

    run_tf_image_resnet(depth=34, filter_scale=1.0)

def test_tf_image_resnet_50():
    global is_pytest_run
    is_pytest_run = True

    run_tf_image_resnet(depth=50, filter_scale=1.0)

def test_tf_image_resnet_101():
    global is_pytest_run
    is_pytest_run = True

    run_tf_image_resnet(depth=101, filter_scale=1.0)

def test_tf_image_resnet_152():
    global is_pytest_run
    is_pytest_run = True

    run_tf_image_resnet(depth=152, filter_scale=1.0)

def run_tf_image_resnet(depth, filter_scale=1.0):
    global is_pytest_run
    model_string = '_d{}_fs{}_'.format(depth, filter_scale)
    test_outputs_dir = 'catamount/frameworks/example_graphs/tensorflow/full_models/image_classification'
    graph_meta = None
    for root, dirs, files in os.walk(test_outputs_dir):
        for filename in files:
            if 'graph{}'.format(model_string) in filename and '.meta' in filename:
                # Take the first graph that we find in the directory
                graph_meta = os.path.join(root, filename)
                break
        if graph_meta is not None:
            break

    if graph_meta is None:
        raise FileNotFoundError('Unable to find model string {} in directory {}'
                                .format(model_string, test_outputs_dir))

    graph = catamount.frameworks.tensorflow.import_graph(graph_meta)
    assert graph.isValid()

    # Manually remove the inference parts of graph
    graph_ops = list(graph._ops_by_name.values())
    for op in graph_ops:
        # Certain ops are only used for inference
        if 'InferenceTower/' in op.name or \
           'InferenceRunner/' in op.name or \
           op.name == 'MergeAllSummariesRunWithOp/Merge/MergeSummary':
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
    output_classes_symbol = utils.getPositiveIntSymbolFromString('out_classes')
    subbatch_size_symbol = utils.getPositiveIntSymbolFromString('subbatch_size')
    image_height_symbol = utils.getPositiveIntSymbolFromString('image_height')
    image_width_symbol = utils.getPositiveIntSymbolFromString('image_width')
    num_in_channels_symbol = utils.getPositiveIntSymbolFromString('num_in_channels')
    graph_iters_symbol = utils.getIntSymbolFromString('graph::iters')
    feature_channels_symbol = utils.getPositiveIntSymbolFromString('feature_channels')

    # Find and replace convolution/pooling dimensions also:
    # Dimension(64 * 2^k): conv/pool feature channels
    base_output_classes = 1000
    base_num_in_channels = 3
    base_feature_channels = 64

    base_image_height = 224
    base_image_width = 224
    base_half_im_height = 112
    half_im_height_symbol = image_height_symbol // 2
    base_half_im_width = 112
    half_im_width_symbol = image_width_symbol // 2
    base_quart_im_height = 56
    quart_im_height_symbol = (image_height_symbol // 2) // 2
    base_quart_im_width = 56
    quart_im_width_symbol = (image_width_symbol // 2) // 2
    base_eighth_im_height = 28
    eighth_im_height_symbol = ((image_height_symbol // 2) // 2) // 2
    base_eighth_im_width = 28
    eighth_im_width_symbol = ((image_width_symbol // 2) // 2) // 2
    base_sixtnth_im_height = 14
    sixtnth_im_height_symbol = (((image_height_symbol // 2) // 2) // 2) // 2
    base_sixtnth_im_width = 14
    sixtnth_im_width_symbol = (((image_width_symbol // 2) // 2) // 2) // 2
    base_small_im_height = 7
    small_im_height_symbol = ((((image_height_symbol // 2) // 2) // 2) // 2) // 2
    base_small_im_width = 7
    small_im_width_symbol = ((((image_width_symbol // 2) // 2) // 2) // 2) // 2

    # TODO (Joel): Add InputQueue ops to avoid manually setting dimensions
    in_deque_op = graph.opsByName['QueueInput/input_deque']
    # print(in_deque_op.debugString())
    out_tensor = in_deque_op._outputs[0]
    for idx, sym in enumerate([subbatch_size_symbol, image_height_symbol, image_width_symbol, num_in_channels_symbol]):
        out_tensor.shape.setDimension(idx, sym)
        out_tensor.shape.dims[idx]._value = None
    out_tensor = in_deque_op._outputs[1]
    out_tensor.shape.setDimension(0, subbatch_size_symbol)

    # Set up a dictionary of placeholders and variables for which we want
    # to make dimensions symbolic. Sift out their dimensions
    bind_dict = { # Placeholders
                  'label': [subbatch_size_symbol],
                  'input': [subbatch_size_symbol, image_height_symbol, image_width_symbol, num_in_channels_symbol],
                }
    # Parameterize all variable tensor dimensions
    for op in graph._ops_by_name.values():
        if isinstance(op, VariableOp):
            op_name_suffix = op.name.split('/')[-1]
            if op_name_suffix == 'W':
                if op._outputs[0].shape.rank == 4:
                    assert 'conv' in op.name
                    new_shape = []
                    for i in range(op._outputs[0].shape.rank):
                        new_shape.append(op._outputs[0].shape.getDimension(i).value)
                    if new_shape[2] % base_feature_channels == 0:
                        in_filters = (new_shape[2] // \
                                      base_feature_channels) * \
                                     feature_channels_symbol
                    elif new_shape[2] == 3:
                        # This is the first convolution on image channels (3)
                        assert op.name == 'conv0/W'
                        in_filters = num_in_channels_symbol
                    else:
                        print('FIX ME: base in filters {}'.format(new_shape[2]))
                        assert 0
                    if new_shape[3] % base_feature_channels == 0:
                        out_filters = (new_shape[3] // \
                                       base_feature_channels) * \
                                       feature_channels_symbol
                    else:
                        print('FIX ME: base out filters {}'.format(new_shape[3]))
                        assert 0
                    new_shape[2] = in_filters
                    new_shape[3] = out_filters
                else:
                    # This is the output layer with output_classes dimension
                    assert op.name == 'linear/W'
                    assert op._outputs[0].shape.rank == 2
                    in_dim = op._outputs[0].shape.getDimension(0).value
                    assert in_dim % base_feature_channels == 0
                    in_dim = (in_dim // base_feature_channels) * \
                             feature_channels_symbol
                    new_shape = [in_dim, output_classes_symbol]
                bind_dict[op.name] = new_shape
                momentum_op_name = '{}/Momentum'.format(op.name)
                momentum_op = graph._ops_by_name[momentum_op_name]
                bind_dict[momentum_op.name] = new_shape
            elif op_name_suffix == 'b':
                # This is the output layer with output_classes dimension
                assert op.name == 'linear/b'
                assert op._outputs[0].shape.rank == 1
                assert op._outputs[0].shape.getDimension(0).value == \
                       base_output_classes
                new_shape = [output_classes_symbol]
                bind_dict[op.name] = new_shape
                momentum_op_name = '{}/Momentum'.format(op.name)
                momentum_op = graph._ops_by_name[momentum_op_name]
                bind_dict[momentum_op.name] = new_shape
            elif op_name_suffix == 'beta' or op_name_suffix == 'gamma' or \
                 op_name_suffix == 'EMA':
                assert op._outputs[0].shape.rank == 1
                in_dim = op._outputs[0].shape.getDimension(0).value
                assert in_dim % base_feature_channels == 0
                in_dim = (in_dim // base_feature_channels) * \
                         feature_channels_symbol
                new_shape = [in_dim]
                bind_dict[op.name] = new_shape
                if op_name_suffix != 'EMA':
                    momentum_op_name = '{}/Momentum'.format(op.name)
                    momentum_op = graph._ops_by_name[momentum_op_name]
                    bind_dict[momentum_op.name] = new_shape

    # Now handle constant values in the graph
    const_dict = {}
    for op in graph._ops_by_name.values():
        if isinstance(op, ConstantOp):
            if op._outputs[0].value is None:
                continue
            if op._outputs[0].shape.rank == 0:
                print('{}'.format(op.debugString()))
                continue
            assert op._outputs[0].shape.rank == 1
            values = op._outputs[0].value.tolist()
            new_values = []
            changed = False
            for value in values:
                if value > 0 and value % base_feature_channels == 0:
                    value = (value // base_feature_channels) * feature_channels_symbol
                    changed = True
                new_values.append(value)
            # HACKY SPECIAL CASE:
            if op.name == 'tower0/gradients/tower0/conv0/Conv2D_grad/Const':
                assert new_values[2] == base_num_in_channels
                new_values[2] = num_in_channels_symbol
            if changed:
                const_dict[op.name] = new_values

    graph.bindConstantValues(const_dict)

    graph.bindShapesAndPropagate(bind_dict, warn_if_ill_defined=(not is_pytest_run), make_symbolic=True)

    # Nice little hack to actually propagate MaximumOp values to outputs
    for op in graph._ops_by_name.values():
        if isinstance(op, MaximumOp):
            if op._inputs[0].value is not None and \
               op._inputs[1].value is not None:
                vmax = np.vectorize(lambda x, y: sympy.Max(x, y))
                out_val = vmax(op._inputs[0].value, op._inputs[1].value)
                op._outputs[0].setValue(out_val)

    graph.bindShapesAndPropagate(bind_dict, warn_if_ill_defined=(not is_pytest_run), make_symbolic=True)

    print('Bound values')

    print(graph)

    bind_subs = {
        graph_iters_symbol: 1,
        output_classes_symbol: base_output_classes,
        subbatch_size_symbol: 32,
        image_height_symbol: base_image_height,
        image_width_symbol: base_image_width,
        num_in_channels_symbol: base_num_in_channels,
        feature_channels_symbol: base_feature_channels,
    }


    correct_params = -1
    correct_flops = -1
    correct_bytes = -1
    correct_footprint = -1
    if depth == 18:
        correct_params = 11689514
        correct_flops = 349684163360
        correct_bytes = 7186222676
        correct_footprint = 2802084304
    elif depth == 34:
        correct_params = 21797674
        correct_flops = 705506994208
        correct_bytes = 11162578644
        correct_footprint = 4368689744
    elif depth == 50:
        correct_params = 25557034
        correct_flops = 790954958112
        correct_bytes = 32896462028
        correct_footprint = 12909734408
    elif depth == 101:
        correct_params = 44549162
        correct_flops = 1506507229472
        correct_bytes = 50026672916
        correct_footprint = 19690293072
    elif depth == 152:
        correct_params = 60192810
        correct_flops = 2222688328992
        correct_bytes = 70967716188
        correct_footprint = 27971880088
    else:
        print('WARN: Tests not defined for depth {}'.format(depth))


    # Calculate parameters
    # NOTE: Need to remove Momentum optimizer parameters and moving average values
    momentum_params = 0
    parameters = 0
    for op_name in sorted(graph.opsByName.keys()):
        op = graph.opsByName[op_name]
        if isinstance(op, VariableOp):
            if "Momentum" in op.name or "EMA" in op.name:
                momentum_params += op.calcModelParameters()
            else:
                parameters += op.calcModelParameters()

    all_weights = graph.calcModelParameters()
    assert (all_weights - momentum_params - parameters) == 0

    # Calculate model parameter count
    resolved_params = parameters.subs(bind_subs)
    try:
        resolved_params = int(resolved_params)
    except:
        print('ERROR: resolved_params should be int, but is {} = {}'.format(
              type(resolved_params), resolved_params))
    assert correct_params < 0 or resolved_params == correct_params, \
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
    assert correct_flops < 0 or resolved_flops == correct_flops, \
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
    assert correct_bytes < 0 or resolved_bytes == correct_bytes, \
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
    assert correct_footprint < 0 or resolved_footprint == correct_footprint, \
           'Incorrect algorithmic footprint: {}'.format(resolved_footprint)
    print('Alg mem footprint: {}\nWith specified dims: {}\n'.format(alg_footprint, resolved_footprint))

    # Calculate algorithmic IO per step
    total_io_footprint = 0
    for op in graph.getPlaceholders():
        total_io_footprint += op.calcAlgFootprint()
    resolved_io_footprint = total_io_footprint.subs(bind_subs)
    print('Alg IO footprint: {}\nWith specified dims: {}\n'.format(total_io_footprint, resolved_io_footprint))

    try: # In case the footprint code is not complete
        # Calculate minimal memory footprint
        print('Alg min mem footprint {}'.format(graph.calcMinFootprint(symbol_subs=bind_subs)))
    except:
        pass


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
    pickle.dump(graph, open('catamount/frameworks/example_graphs/tensorflow/full_models/image_classification/graph_image_resnet_d{}_fs{}.p'.format(depth, filter_scale), 'wb'))

    if is_pytest_run:
        return

    print('\n\n======= Algorithmic graph-level analytics: =======')

    feature_channel_dims = [32, 48, 64, 96, 128]

    bind_subs.pop(feature_channels_symbol)
    resolved_params = parameters.subs(bind_subs)

    print('Symbol associations: {}\n'.format(bind_subs))

    print('Algorithmic Flops by feature channels, params, and per-batch-sample:')
    resolved_flops = alg_flops.subs(bind_subs)
    for features_dim in feature_channel_dims:
        graph_params = resolved_params.subs({feature_channels_symbol: features_dim})
        graph_flops = resolved_flops.subs({feature_channels_symbol: features_dim})
        graph_flops_per_sample = float(graph_flops) / \
                                 bind_subs[subbatch_size_symbol]
        print('{}\t{}\t{}\t{}'.format(features_dim, graph_params, graph_flops,
                                      int(graph_flops_per_sample)))

    print('\nAlgorithmic bytes accessed by feature channels, params:')
    resolved_bytes = alg_bytes.subs(bind_subs)
    for features_dim in feature_channel_dims:
        graph_params = resolved_params.subs({feature_channels_symbol: features_dim})
        graph_bytes = resolved_bytes.subs({feature_channels_symbol: features_dim})
        print('{}\t{}\t{}'.format(features_dim, graph_params, graph_bytes))

    print('\nAlgorithmic total memory footprint by feature channels, params:')
    resolved_footprint = alg_footprint.subs(bind_subs)
    for features_dim in feature_channel_dims:
        graph_params = resolved_params.subs({feature_channels_symbol: features_dim})
        graph_footprint = resolved_footprint.subs({feature_channels_symbol: features_dim})
        print('{}\t{}\t{}'.format(features_dim, graph_params, graph_footprint))

    print('\nAlgorithmic minimal memory footprint by feature channels, params:')
    full_subs = dict(bind_subs)
    for features_dim in feature_channel_dims:
        graph_params = resolved_params.subs({feature_channels_symbol: features_dim})
        full_subs[feature_channels_symbol] = features_dim
        graph_min_foot = graph.calcMinimalFootprint(symbol_subs=full_subs)
        print('{}\t{}\t{}'.format(features_dim, graph_params, graph_min_foot))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, default=50, help='ResNet model size')
    parser.add_argument('--filter_scale', type=float, default=1.0,
                        help='ResNet model filter scale')
    args = parser.parse_args()

    run_tf_image_resnet(depth=args.depth,
                        filter_scale=args.filter_scale)

