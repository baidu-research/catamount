import os
import struct
import tensorflow as tf


# To import graphs that use various TF sublibraries...
try:
    import tensorflow.contrib.mpi_collectives as mpi
except Exception as exc:
    print('WARN: Cannot import tensorflow.contrib.mpi_collectives...' \
          ' not built?')

try:
    import tensorswift as ts
except Exception as exc:
    print('WARN: Cannot import tensorswift... not installed?')

from tensorflow.core.framework import types_pb2

import catamount
from catamount.graph import *
from catamount.ops import *
from catamount.tensors.tensor import *

# Tools to import Tensorflow MetaGraphs into Catamount format

class TFRestoreOp(Op):
    ''' A designated op type for Tensorflow model saver restore ops.
        The intent of this op is to identify model saver ops that
        should be removed from the graph before returning it to the
        import_graph() caller.
    '''
    def __init__(self, name):
        super(TFRestoreOp, self).__init__(name)

class TFSaveOp(Op):
    ''' A designated op type for Tensorflow model saver save ops.
        The intent of this op is to identify model saver ops that
        should be removed from the graph before returning it to the
        import_graph() caller.
    '''
    def __init__(self, name):
        super(TFSaveOp, self).__init__(name)


TF_OP_TO_CATAMOUNT = {
    'Add': AddOp,
    'AddN': AddNOp,
    'All': ReduceOp,
    'ArgMax': ReduceOp,
    'Any': ReduceOp,
    'ApplyGradientDescent': ApplyGradientDescentOp,
    'ApplyMomentum': ApplyMomentumOp,
    'Assign': AssignOp,
    'AssignAdd': AddOp, # Here, TF reuses the input tensor for output
    'AssignSub': SubOp, # Here, TF reuses the input tensor for output
    'BatchMatMul': BatchMatMulOp,
    'BiasAdd': AddOp, # Here, TF special-case for 1D bias input
    'BiasAddGrad': ReduceOp, # Here, TF special-case to backprop bias
    'BroadcastGradientArgs': BroadcastGradientArgsOp,
    'Cast': CastOp,
    'ConcatV2': ConcatOp,
    'ConcatOffset': ConcatOffsetOp,
    'Const': ConstantOp,
    'ControlTrigger': NoOp, # Ops added for synchronization only
    'Conv2DBackpropFilter': Conv2DGradFilterOp,
    'Conv2DBackpropInput': Conv2DGradInputOp,
    'Conv2D': Conv2DOp,
    'DynamicStitch': DynamicStitchOp,
    'Enter': EnterOp,
    'Equal': EqualOp,
    'Erf': ErfOp,
    'Exit': ExitOp,
    'Exp': ExpOp,
    'ExpandDims': ExpandDimsOp,
    'Fill': FillOp,
    'Floor': FloorOp,
    'FloorDiv': FloorDivOp,
    'FloorMod': FloorModOp,
    'FusedBatchNorm': FusedBatchNormOp,
    'FusedBatchNormGrad': FusedBatchNormGradOp,
    'Gather': GatherOp,
    # Same as TF Gather, but adds additional input[2] = axis of gather
    'GatherV2': GatherOp,
    'Greater': GreaterOp,
    'GreaterEqual': GreaterEqualOp,
    'Identity': IdentityOp,
    'InTopKV2': InTopKOp,
    'InvertPermutation': InvertPermutationOp,
    'Less': LessOp,
    'LessEqual': LessEqualOp,
    'ListDiff': ListDiffOp,
    'Log': LogOp,
    'Log1p': Log1pOp,
    'LogSoftmax': SoftmaxOp,
    'LogUniformCandidateSampler': CandidateSamplerOp,
    'LogicalAnd': LogicalAndOp,
    'LogicalOr': LogicalOrOp,
    'LogicalNot': LogicalNotOp,
    'LoopCond': LoopConditionOp,
    'L2Loss': L2LossOp,
    'MatMul': MatMulOp,
    'Max': ReduceOp,
    'Maximum': MaximumOp,
    'MaxPool': MaxPoolOp,
    'MaxPoolGrad': MaxPoolGradOp,
    'Min': ReduceOp,
    'Mean': ReduceOp,
    'Merge': MergeOp,
    'Minimum': MinimumOp,
    'MPIAllgather': AllgatherOp,
    'MPIAllreduce': AllreduceOp,
    # tf.contrib.mpi_collectives.MPIInit has no compute graph function
    'MPIInit': NoOp,
    # tf.contrib.mpi_collectives.MPISize behaves like a placeholder
    'MPISize': PlaceholderOp,
    'Mul': MulOp,
    'Multinomial': MultinomialOp,
    'Neg': NegOp,
    'NextIteration': NextIterationOp,
    'NoOp': NoOp, # Ignore no-ops
    'NotEqual': NotEqualOp,
    'OneHot': OneHotOp,
    'OnesLike': NumLikeOp,
    'Pack': PackOp,
    'Pad': PadOp,
    'Placeholder': PlaceholderOp,
    'PlaceholderWithDefault': IdentityOp,
    'PreventGradient': PreventGradientOp,
    'Prod': ReduceOp,
    'Pow': PowOp,
    'RandomUniformInt': RandomInitializerOp,
    'RandomUniform': RandomInitializerOp,
    'RandomStandardNormal': RandomInitializerOp,
    'Range': RangeOp,
    'Rank': RankOp,
    'RealDiv': BasePointwiseOp,
    'Reciprocal': ReciprocalOp,
    'Relu': ReluOp,
    'ReluGrad': ReluGradOp,
    'Reduce': ReduceOp,
    'RefEnter': EnterOp,
    'Reshape': ReshapeOp,
    'RestoreV2': TFRestoreOp, # Identify Restore ops for removal
    'ReverseSequence': ReverseSequenceOp,
    'Rsqrt': RsqrtOp,
    'RsqrtGrad': RsqrtGradOp,
    'SaveV2': TFSaveOp, # Identify Saver ops for removal
    'Scatter': ScatterOp,
    'ScatterSub': ScatterUpdateOp,
    'Select': SelectOp,
    'Shape': ShapeOp,
    'ShapeN': ShapeOp, # ShapeOp takes multiple inputs like TF ShapeN
    'Slice': SliceOp,
    'Sigmoid': SigmoidOp,
    'SigmoidGrad': SigmoidGradOp,
    'Size': SizeOp,
    'Softmax': SoftmaxOp,
    'SparseSoftmaxCrossEntropyWithLogits': SparseSoftmaxCrossEntropyWithLogitsOp,
    'Split': SplitOp,
    'SplitV': SplitOp,
    'Sqrt': SqrtOp,
    'SqrtGrad': SqrtGradOp,
    'StridedSlice': StridedSliceOp,
    'StridedSliceGrad': StridedSliceGradOp,
    'Sub': SubOp,
    'Sum': ReduceOp,
    'SquaredDifference': SquaredDifferenceOp,
    'Square': SquareOp,
    'Squeeze': SqueezeOp,
    'StackPopV2': StackPopOp,
    'StackPushV2': StackPushOp,
    'StackV2': StackOp,
    'StopGradient': StopGradientOp,
    'Switch': SwitchOp,
    'Tanh': TanhOp,
    'TanhGrad': TanhGradOp,
    'TensorArrayV3': TensorArrayOp,
    'Tile': TileOp,
    'Transpose': TransposeOp,
    'TruncatedNormal': RandomInitializerOp,
    'Unpack': UnpackOp,
    'UnsortedSegmentSum': UnsortedSegmentSumOp,
    'VariableV2': VariableOp,
    'Where': WhereOp,
    'ZerosLike': NumLikeOp,
}

TF_OP_TO_CATAMOUNT_REDUCE = {
    # 'All': None,
    # 'ArgMax': None,
    # 'Any': None,
    'BiasAddGrad': 'sum',
    # 'Max': ReduceOp,
    # 'Min': ReduceOp,
    # 'Mean': ReduceOp,
    'Prod': 'product',
    # 'Reduce': ReduceOp,
    'Sum': 'sum',
}

# TODO (Joel): Prioritize these ops:
# -- NMT
# -- Speech
# -- Others

# TODO (Joel): These are required for accurate counts, but we can hack
# TensorArrayGatherV3 # Same shape output as TensorArray input
# TensorArrayGradV3
# TensorArrayReadV3 # Same shape output as TensorArray input sliced on first dim
# TensorArrayScatterV3
# TensorArraySizeV3 # Input 1: Dim0 of tensor input to TensorArray
# TensorArrayWriteV3
# QueueDequeueV2
# QueueEnqueueV2
# Stack
# StackPop
# StackPopV2 # Same shape output as StackPush input
# StackPush
# StackPushV2
# StackV2

# TODO (Joel): Low priority. Counts are accurate without
# Assert
# FIFOQueueV2
# FilterDataset
# GroupByWindowDataset
# HashTableV2
# InitializeTableFromTextFileV2
# Iterator
# IteratorGetNext
# IteratorToStringHandle
# LookupTableFindV2
# MakeIterator
# MapDataset
# MergeSummary
# Pad
# ParallelMapDataset
# PrefetchDataset
# QueueCloseV2
# QueueSizeV2
# RangeDataset
# Round
# ScalarSummary
# ShuffleDataset
# SkipDataset
# Stage
# TextLineDataset
# TruncatedNormal
# Unstage
# ZipDataset

TF_DTYPE_TO_CATAMOUNT = {
    tf.bool: DataType.bool,
    tf.int32: DataType.int32,
    tf.int64: DataType.int64,
    tf.uint8: DataType.uint8,
    tf.float32: DataType.float32,
    tf.float64: DataType.float64,
    tf.string: DataType.string,
}

def tf_shape_to_catamount(tf_shape):
    dims = None
    if tf_shape is not None and tf_shape.ndims is not None:
        dims = []
        if tf_shape.dims is not None and len(tf_shape.dims) > 0:
            for dim in tf_shape.dims:
                dims.append(dim.value)
    return TensorShape(dims)

def load_tf_session(tf_filename):
    if not os.path.exists(tf_filename):
        raise FileNotFoundError('{}'.format(tf_filename))
    saver = tf.train.import_meta_graph(tf_filename)
    sess = tf.Session()
    try:
        tf_model_name = tf_filename.replace('.meta', '')
        saver.restore(sess, tf_model_name)
    except Exception:
        print('WARN: Cannot find checkpoint data {}, trying to proceed'
              .format('{}.data-?-of-?'.format(tf_filename)))
    return sess

def import_graph(tf_filename):
    sess = load_tf_session(tf_filename)
    catamount_graph = construct_catamount_graph(sess, sess.graph)
    # Clean up TF bits to avoid problems with successive graph loads
    sess.close()
    tf.reset_default_graph()
    return catamount_graph

def try_to_read_value_from_tensor(tf_sess, tf_op):
    # Simple function to try to read tensors out of a session. For
    # testing purposes only.
    try:
        with tf_sess.as_default():
            value = tf_op.outputs[0].eval()
    except:
        raise NotImplementedError('Exception')
    return value

unpack_types = [ types_pb2.DT_INT32,
                 types_pb2.DT_INT64,
                 types_pb2.DT_FLOAT ]
unpack_strs =  { types_pb2.DT_INT32: 'i',
                 types_pb2.DT_INT64: 'q',
                 types_pb2.DT_FLOAT: 'f' }

def get_value_from_proto(tf_op, value_proto):
    if value_proto.dtype == types_pb2.DT_BOOL:
        return value_proto.bool_val
    elif value_proto.dtype == types_pb2.DT_INT32:
        return value_proto.int_val
    elif value_proto.dtype == types_pb2.DT_INT64:
        return value_proto.int64_val
    elif value_proto.dtype == types_pb2.DT_FLOAT:
        return value_proto.float_val
    elif value_proto.dtype == types_pb2.DT_DOUBLE:
        return value_proto.double_val
    elif value_proto.dtype == types_pb2.DT_STRING:
        return value_proto.string_val
    else:
        raise NotImplementedError('Op {}: Unhandled dtype: {}'
                                  .format(tf_op.name, value_proto.dtype))

def get_const_value_from_op(tf_sess, tf_op):
    assert tf_op.type == 'Const'
    assert len(tf_op.outputs) == 1
    tf_op.outputs[0].shape.assert_is_fully_defined()

    value_proto = tf_op.get_attr('value')
    # Sure wish there was a better way to recover these through TF...
    if value_proto.dtype == types_pb2.DT_BOOL or \
       value_proto.dtype == types_pb2.DT_INT32 or \
       value_proto.dtype == types_pb2.DT_INT64:
        if tf_op.outputs[0].shape.ndims == 0 or \
           tf_op.outputs[0].shape.num_elements() == 1:
            value = get_value_from_proto(tf_op, value_proto)
            assert len(value) == 1, \
                'Op: {} value: {}'.format(tf_op.name, value)
            value = value[0]
        else:
            s = struct.Struct(unpack_strs[value_proto.dtype])
            it = s.iter_unpack(value_proto.tensor_content)
            value = np.array([x[0] for x in it])
            assert len(value) == tf_op.outputs[0].shape.num_elements(), \
                'Op: {}, value: {}'.format(tf_op.name, value)
    elif value_proto.dtype == types_pb2.DT_FLOAT or \
         value_proto.dtype == types_pb2.DT_DOUBLE:
        if tf_op.outputs[0].shape.ndims == 0 or \
           tf_op.outputs[0].shape.num_elements() == 1:
            value = get_value_from_proto(tf_op, value_proto)
            assert len(value) == 1, \
                'Op: {} value: {}'.format(tf_op.name, value)
            value = value[0]
        else:
            s = struct.Struct(unpack_strs[value_proto.dtype])
            it = s.iter_unpack(value_proto.tensor_content)
            value = [x[0] for x in it]
            if len(value) == 0:
                value = get_value_from_proto(tf_op, value_proto)
                if len(value) == 0:
                    print('WARN: Unable to read op {} value from proto'
                          .format(tf_op.name))
                    value = [None]
                np_shape = tf_op.outputs[0].shape.as_list()
                value = np.full(np_shape, value[0], dtype=float)
            else:
                value = np.array(value)
            tf_op_shape_list = tf_op.outputs[0].shape.as_list()
            if list(value.shape) != tf_op_shape_list:
                try:
                    value = np.reshape(value, tf_op_shape_list)
                except ValueError as err:
                    print('{}:\nShape mismatch. Value {}, shapes '\
                          '{} != {}'.format(tf_op.name, value,
                                            list(value.shape),
                                            tf_op_shape_list))
                    raise err
            assert list(value.shape) == tf_op.outputs[0].shape.as_list(), \
                'Op: {}, value: {}, len(value): {}, out_shape: {}' \
                .format(tf_op.name, value, list(value.shape),
                        tf_op.outputs[0].shape.as_list())
    elif value_proto.dtype == types_pb2.DT_STRING:
        if tf_op.outputs[0].shape.ndims == 0 or \
           tf_op.outputs[0].shape.num_elements() == 1:
            value = get_value_from_proto(tf_op, value_proto)
            assert len(value) == 1, \
                'Op: {} value: {}'.format(tf_op.name, value)
            value = value[0].decode('utf-8')
        else:
            value = []
            for i in range(tf_op.outputs[0].shape.num_elements()):
                value.append(value_proto.string_val[i].decode('utf-8'))
            value = np.array(value)
            assert list(value.shape) == tf_op.outputs[0].shape.as_list(), \
                'Op: {}, value: {}'.format(tf_op.name, value)
    else:
        raise NotImplementedError('Other TF op {} dtype to handle {}'
                                  .format(tf_op.name, value_proto.dtype))
    return value

def get_transpose_attributes_from_op(tf_sess, tf_op, op):
    if tf_op.get_attr('transpose_a'):
        op.setTransposeInput(0, True)
    if tf_op.get_attr('transpose_b'):
        op.setTransposeInput(1, True)

def get_slice_attributes_from_op(tf_sess, tf_op, op):
    op.setBeginMask(tf_op.get_attr('begin_mask'))
    op.setEllipsisMask(tf_op.get_attr('ellipsis_mask'))
    op.setEndMask(tf_op.get_attr('end_mask'))
    op.setNewAxisMask(tf_op.get_attr('new_axis_mask'))
    op.setShrinkAxisMask(tf_op.get_attr('shrink_axis_mask'))

def get_split_attributes_from_op(tf_sess, tf_op, op):
    op.setNumSplit(tf_op.get_attr('num_split'))

def get_conv_attributes_from_op(tf_sess, tf_op, op):
    op.setDataFormat(tf_op.get_attr('data_format').decode('utf-8'))
    op.setStrides(tf_op.get_attr('strides'))
    if isinstance(op, (Conv2DOp, Conv2DGradInputOp)):
        op.setDilations(tf_op.get_attr('dilations'))

def get_batch_norm_attributes_from_op(tf_sess, tf_op, op):
    op.setDataFormat(tf_op.get_attr('data_format').decode('utf-8'))

def get_pool_attributes_from_op(tf_sess, tf_op, op):
    op.setDataFormat(tf_op.get_attr('data_format').decode('utf-8'))
    op.setKSize(tf_op.get_attr('ksize'))
    op.setStrides(tf_op.get_attr('strides'))

def get_axis_attribute_from_op(tf_sess, tf_op, op):
    op.setAxis(tf_op.get_attr('axis'))

def get_batch_matmul_attributes_from_op(tf_sess, tf_op, op):
    op.setAdjointX(tf_op.get_attr('adj_x'))
    op.setAdjointY(tf_op.get_attr('adj_y'))

def get_enter_frame_name_from_op(tf_sess, tf_op, op):
    op.setFrameName(tf_op.get_attr('frame_name').decode('utf-8'))

def get_squeeze_dims_from_op(tf_sess, tf_op, op):
    op.setSqueezeDims(tf_op.get_attr('squeeze_dims'))

def get_keep_dims_from_op(tf_sess, tf_op, op):
    keep_dims = False
    try:
        keep_dims = tf_op.get_attr('keep_dims')
    except:
        pass
    if keep_dims:
        op.setKeepDims(tf_op.get_attr('keep_dims'))

def parse_tf_op_attributes_into_op(tf_sess, tf_op, op):
    # tf_op.op_def is the parameterization for protobuf
    # tf_op.node_def contains the arguments from protobuf to apply to op
    # instance
    if isinstance(op, ConstantOp):
        # For ConstantOps, we may need their value to resolve tensor shapes
        # for downstream ops. Collect and set in the op
        op.outputs[0].setValue(get_const_value_from_op(tf_sess, tf_op))

    elif isinstance(op, MatMulOp):
        # MatMuls may specify transposes as attributes to the op
        get_transpose_attributes_from_op(tf_sess, tf_op, op)

    elif isinstance(op, StridedSliceOp):
        # StridedSliceOps can have mask attributes
        get_slice_attributes_from_op(tf_sess, tf_op, op)

    elif isinstance(op, SplitOp):
        get_split_attributes_from_op(tf_sess, tf_op, op)

    elif isinstance(op, (Conv2DOp, Conv2DGradFilterOp,
                         Conv2DGradInputOp)):
        get_conv_attributes_from_op(tf_sess, tf_op, op)

    elif isinstance(op, (FusedBatchNormOp, FusedBatchNormGradOp)):
        get_batch_norm_attributes_from_op(tf_sess, tf_op, op)

    elif isinstance(op, PoolBaseOp):
        get_pool_attributes_from_op(tf_sess, tf_op, op)

    elif isinstance(op, (PackOp, UnpackOp)):
        get_axis_attribute_from_op(tf_sess, tf_op, op)

    elif isinstance(op, BatchMatMulOp):
        get_batch_matmul_attributes_from_op(tf_sess, tf_op, op)

    elif isinstance(op, EnterOp):
        get_enter_frame_name_from_op(tf_sess, tf_op, op)

    elif isinstance(op, SqueezeOp):
        get_squeeze_dims_from_op(tf_sess, tf_op, op)

    elif isinstance(op, ReduceOp):
        get_keep_dims_from_op(tf_sess, tf_op, op)

    # print(tf_op.op_def)
    # print(tf_op.node_def)

def construct_catamount_graph(tf_sess, tf_graph):
    graph = Graph()
    tensors = {}
    op_inputs = {}
    ctrl_frames = {}
    all_stack_ops = []
    for tf_op in tf_graph._nodes_by_name.values():
        if tf_op.type in TF_OP_TO_CATAMOUNT.keys():
            # Map to Catamount op type
            catamount_type = TF_OP_TO_CATAMOUNT[tf_op.type]
        else:
            print('WARN: Unknown op type: {} (op: {})'
                  .format(tf_op.type, tf_op.name))
            catamount_type = UnknownOp

        # Create the Catamount internal op
        op = catamount_type(tf_op.name)

        if catamount_type == ReduceOp:
            reduce_op = None
            if tf_op.type in TF_OP_TO_CATAMOUNT_REDUCE:
                reduce_op = TF_OP_TO_CATAMOUNT_REDUCE[tf_op.type]
            else:
                print('WARN: Reduce may set reduction op: {}'.format(tf_op.type))
            if reduce_op is not None:
                op.setReductionOp(reduce_op)
            if tf_op.type == 'BiasAddGrad':
                op.setAxes(0)

        if catamount_type == ScatterUpdateOp:
            print('WARN: ScatterUpdate may set update op: {}'.format(tf_op.type))

        # Create the output tensors for this op
        for i in range(len(tf_op.outputs)):
            tf_tensor = tf_op.outputs[i]

            tf_dtype = tf_tensor.dtype.base_dtype
            if tf_dtype in TF_DTYPE_TO_CATAMOUNT.keys():
                catamount_dtype = TF_DTYPE_TO_CATAMOUNT[tf_dtype]
            else:
                print('WARN: Unknown dtype {} for tensor {}'
                      .format(tf_tensor.dtype, tf_tensor))
                catamount_dtype = None

            out_tens = Tensor(tf_tensor.name,
                tf_shape_to_catamount(tf_tensor.shape), catamount_dtype)
            tensors[out_tens.name] = out_tens
            op.addOutput(out_tens)

        # Track the input tensor names to connect them in next phase
        op_inputs[op.name] = []
        if tf_op.type == 'Split':
            # TF Split op has different interface than Catamount. Need to add the
            # size_splits tensor to match the Catamount interface (input[1])
            assert len(tf_op.inputs) == 2
            op_inputs[op.name].append(tf_op.inputs[1].name)
            # Signal to Catamount to use the num_split attribute by setting
            # size_splits equal to a scalar constant of value 0
            size_splits = catamount.constant('{}_size_splits'.format(op.name),
                                         out_shape=[], value=0, graph=graph)
            tensors[size_splits.name] = size_splits
            op_inputs[op.name].append(size_splits.name)
            op_inputs[op.name].append(tf_op.inputs[0].name)
        else:
            for i in range(len(tf_op.inputs)):
                op_inputs[op.name].append(tf_op.inputs[i].name)

        # Get the tf_op's attributes and set them as necessary
        parse_tf_op_attributes_into_op(tf_sess, tf_op, op)

        if isinstance(op, EnterOp):
            frame_name = op.getFrameName()
            if frame_name not in ctrl_frames:
                ctrl_frames[frame_name] = ContextFrame(frame_name)
            ctrl_frames[frame_name].addEnterOp(op)
        elif isinstance(op, StackOp):
            all_stack_ops.append(op)

        graph.addOp(op)

    # Hook up all the op inputs to the ops that generate them
    for op_name in op_inputs.keys():
        op = graph.opsByName[op_name]
        for in_tensor in op_inputs[op_name]:
            assert in_tensor in tensors.keys(), \
                   'Unknown input tensor {}'.format(in_tensor)
            graph.addInputToOp(op, tensors[in_tensor])

    # Propagate stack pointers for StackOps. These ops always occur as a
    # series of ops. The StackOp is first, and propagates its outputs to
    # (optionally) EnterOps, and then to StackPush and StackPop ops. The
    # StackPush and StackPop ops need to get the pointer for the stack
    # created for the StackOp
    for stack_op in all_stack_ops:
        # Traverse out tensor to find all StackPush and StackPop
        out_tensor = stack_op.outputs[0]
        for cons_op in out_tensor.consumers.values():
            while not isinstance(cons_op, BaseStackOp):
                assert(isinstance(cons_op, (EnterOp, SwitchOp)))
                assert(len(cons_op.outputs[0].consumers) == 1)
                cons_ops = list(cons_op.outputs[0].consumers.values())
                cons_op = cons_ops[0]
            cons_op.setStack(stack_op.getStack())
            assert(cons_op.getStack() is not None)

    # Remove TF variable initialization (Assign) ops
    # These are not necessary to fully specify the graph
    assign_ops = set()
    for op in graph.opsByName.values():
        if isinstance(op, AssignOp):
            assign_ops.add(op)
    op_types = set()
    for assign_op in assign_ops:
        assert isinstance(assign_op.inputs[0].producer, VariableOp)
        # assert isinstance(assign_op.inputs[1].producer, ConstantOp)
        my_ancestors = set()
        my_frontier = set()
        my_frontier.add(assign_op)
        while len(my_frontier) > 0:
            next_op = my_frontier.pop()
            for in_tensor in next_op.inputs:
                if not isinstance(in_tensor.producer, VariableOp):
                    my_frontier.add(in_tensor.producer)
            my_ancestors.add(next_op)
            if len(my_ancestors) > 100:
                break
        if len(my_ancestors) <= 8:
            op_types.update(set(type(op) for op in my_ancestors))
            for next_op in my_ancestors:
                graph.removeOp(next_op)
        else:
            print('WARN: Unable to remove: {}'.format(assign_op.debugString()))
            print('    COUNT: {}'.format(len(my_ancestors)))
    assert graph.isValid()

    # Remove any Tensorflow model saver ops from the graph. These ops
    # always occur as a series of 6 ops:
    # 1) Three ConstOps that define the (A) the name of the model, (B) the
    #    names of saved tensors, and (C) the sizes/shapes of saved tensors.
    # 2) Save and Restore ops, which takes the above inputs 0-2
    # 3) An AssignOp, which takes the above reference, and if appropriate,
    #    loads tensor data from the checkpoint and assigns the ref to it.
    # 4) A control dependency op (IdentityOp) that takes the model name
    #    op as input and has no outputs
    ops_to_remove = set()
    saver_ops = []
    model_name_ops = set()
    for op in graph.opsByName.values():
        if isinstance(op, (TFRestoreOp, TFSaveOp)):
            saver_ops.append(op)
            ops_to_remove.add(op)
            if op.inputs[0].producer not in model_name_ops:
                model_name_ops.add(op.inputs[0].producer)
    for saver_op in saver_ops:
        if isinstance(saver_op, TFRestoreOp):
            assert len(saver_op.inputs) == 3
        else:
            assert isinstance(saver_op, TFSaveOp)
            # First 3 inputs are config inputs
            assert len(saver_op.inputs) >= 3
            assert len(saver_op.outputs) == 0

        # Get input ops and trace back to through consts and idents
        parent_ops_to_trace_and_remove = []
        for idx in range(3):
            in_tensor = saver_op.inputs[idx]
            input_op = in_tensor.producer
            ops_to_remove.add(input_op)
            if len(input_op.inputs) > 0:
                assert isinstance(input_op, (IdentityOp, UnknownOp)), \
                   'Not ident or unk: {}'.format(input_op.debugString())
                parent_ops_to_trace_and_remove.append(input_op)
            else:
                assert isinstance(input_op, (ConstantOp, IdentityOp)), \
                   'Not const: {}'.format(input_op.debugString())
        while len(parent_ops_to_trace_and_remove) > 0:
            parent_op = parent_ops_to_trace_and_remove.pop(0)
            for in_tensor in parent_op.inputs:
                input_op = in_tensor.producer
                ops_to_remove.add(input_op)
                if len(input_op.inputs) > 0:
                    assert isinstance(input_op, (IdentityOp, UnknownOp)), \
                        'Not ident or unk: {}'.format(input_op.debugString())
                    parent_ops_to_trace_and_remove.append(input_op)
                else:
                    assert isinstance(input_op, ConstantOp), \
                        'Not const: {}'.format(input_op.debugString())

        if isinstance(saver_op, TFRestoreOp):
            assert len(saver_op.outputs) >= 1
            # Restore ops can package all tensors together into a single
            # op, so need to traverse all outputs to their assign ops
            child_ops_to_trace_and_remove = []
            for out_tensor in saver_op.outputs:
                assert len(out_tensor.consumers) == 1
                output_op = list(out_tensor.consumers.values())[0]
                ops_to_remove.add(output_op)
                if isinstance(output_op, AssignOp):
                    assert len(output_op.outputs) == 1
                    assert len(output_op.outputs[0].consumers) == 0
                else:
                    assert isinstance(output_op, IdentityOp)
                    child_ops_to_trace_and_remove.append(output_op)
            while len(child_ops_to_trace_and_remove) > 0:
                child_op = child_ops_to_trace_and_remove.pop(0)
                ops_to_remove.add(child_op)
                for child_tens in child_op.outputs:
                    for next_op in child_tens.consumers.values():
                        assert isinstance(next_op, (AssignOp, UnknownOp))
                        child_ops_to_trace_and_remove.append(next_op)
    model_name_consumers = []
    for model_name_op in model_name_ops:
        assert len(model_name_op.outputs) == 1
        model_name_consumers.extend(
            model_name_op.outputs[0].consumers.values())
    for saver_op in model_name_consumers:
        if saver_op not in ops_to_remove:
            # Only other op to catch is the control dependency op, which
            # is an IdentityOp and has no outputs
            if isinstance(saver_op, IdentityOp):
                assert len(saver_op.outputs) == 1
                assert len(saver_op.outputs[0].consumers) == 0
            else:
                print('WARN: Unknown model_name_consumer: {}'
                      .format(saver_op.debugString()))
            ops_to_remove.add(saver_op)
    for op in ops_to_remove:
        graph.removeOp(op)

    assert graph.isValid()

    # Traverse the graph to find subgraph ops, such as loops
    # NOTES:
    #  1) TF while loops are controlled by a LoopConditionOp, which gates
    #     all the SwitchOps that allow a loop iteration to proceed. The
    #     inputs to a LoopConditionOp can be part of the condition function
    #     passed to tf.while_loop. However, the condition function cannot
    #     create side-effects (which is an important observation for
    #     identifying the condition subgraph).
    #  2) The condition subgraph is defined as all inputs to the while loop
    #     that are not updated during the loop body and outputs of MergeOps
    #     that are used to evaluate the loop condition function.
    #  3) Loops create a loop-iteration versioning context for each variable
    #     that is explicitly input into the while condition or body
    #     functions (but NOT variables/tensors that are accessed locally or
    #     globally for evaluating the condition).
    #  4) The body of the loop is all ops that occur between any IdentityOp
    #     and any NextIterationOp from the variable contexts for the loop.
    #  Final) Note that TF while loops can have nested while loops or other
    #     control flow blocks, so we need to design this recursively.
    control_ops = []
    # Find the ops that will require subgraph designations (i.e., control)
    for op_name, op in graph.opsByName.items():
        if op.isControlOp():
            control_ops.append(op)
    assert len(control_ops) == len(ctrl_frames)
    for ctrl_op in control_ops:
        # Get all ops for the loop condition value calculation (1 and 2),
        # the variable contexts (3), and the loop body (4). Extract these
        # into a subgraph.
        ctrl_block_frame = None
        subgraph_ops = [ctrl_op]
        visited_ops = set(subgraph_ops)
        enter_ops = set()
        exit_ops = set()
        frontier_ops = []
        for out_tensor in ctrl_op.outputs:
            for consumer in out_tensor.consumers.values():
                assert isinstance(consumer, SwitchOp)
            frontier_ops.extend(out_tensor.consumers.values())

        # A) Traverse backward from SwitchOps to MergeOps and EnterOps,
        #    and NextIterationOps. Stop at the LoopConditionOp, and any
        #    NextIterationOps and EnterOps. Add MergeOps to the frontier
        #    to traverse forward from them.
        bwd_frontier_ops = list(frontier_ops)
        while len(bwd_frontier_ops) > 0:
            next_op = bwd_frontier_ops.pop(0)
            if next_op in visited_ops:
                continue
            assert not next_op.isControlOp(), \
                'Catamount Framework(TF): Should be no up-stream control blocks!'
            if isinstance(next_op, EnterOp):
                # Add EnterOps to the frontier and visited by looking up all
                # the EnterOps associated with their context frame. Will
                # traverse forward from them to ExitOps or MergeOps
                if ctrl_block_frame is None:
                    ctx_frame = next_op._context_frame
                else:
                    assert ctrl_block_frame == next_op._context_frame
                for enter_op in ctx_frame._enter_ops.values():
                    if enter_op not in frontier_ops:
                        frontier_ops.append(enter_op)
                        enter_ops.add(enter_op)
                # Do not traverse past EnterOps
                continue
            elif isinstance(next_op, NextIterationOp):
                visited_ops.add(next_op)
                # Do not traverse past NextIterationOps
                continue
            elif isinstance(next_op, MergeOp):
                visited_ops.add(next_op)
                frontier_ops.append(next_op)
            for in_tensor in next_op.inputs:
                visited_ops.add(next_op)
                bwd_frontier_ops.append(in_tensor.producer)

        # B) Traverse forward to get the SwitchOps, ExitOps, IdentityOps,
        #    body, NextIterationOps.
        fwd_frontier_ops = []
        for frontier_op in frontier_ops:
            for out_tensor in frontier_op.outputs:
                fwd_frontier_ops.extend(out_tensor.consumers.values())
        while len(fwd_frontier_ops) > 0:
            next_op = fwd_frontier_ops.pop(0)
            if next_op in visited_ops or next_op in enter_ops:
                continue
            if isinstance(next_op, ExitOp):
                # Do not traverse past ExitOps
                exit_ops.add(next_op)
                continue
            if next_op.isControlOp():
                raise NotImplementedError(
                    'Catamount Framework(TF): Need nested control blocks')
            visited_ops.add(next_op)
            for out_tensor in next_op.outputs:
                fwd_frontier_ops.extend(out_tensor.consumers.values())

        # Finally, create a ControlBlockOp (subgraph) with the main control
        # node as the ctrl_op, and add the ControlBlockOp to the Catamount graph
        # (which will move the graph ops into the subgraph)
        ctrl_block_op = ControlBlockOp('{}_block'.format(ctrl_op.name),
                                       ctrl_op, visited_ops, enter_ops,
                                       exit_ops)
        ctrl_block_op.setContextFrame(ctrl_block_frame)
        graph.addOp(ctrl_block_op)

    return graph
