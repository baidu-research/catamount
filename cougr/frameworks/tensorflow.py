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

import cougr
from cougr.graph import *
from cougr.ops import *
from cougr.tensors.tensor import *

# Tools to import Tensorflow MetaGraphs into CouGr format

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


TF_OP_TO_COUGR = {
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
    'ListDiff': ListDiffOp,
    'Log': LogOp,
    'Log1p': Log1pOp,
    'LogUniformCandidateSampler': CandidateSamplerOp,
    'LogicalAnd': LogicalAndOp,
    'LogicalOr': LogicalOrOp,
    'LogicalNot': LogicalNotOp,
    'LoopCond': LoopConditionOp,
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
    'PreventGradient': PreventGradientOp,
    'Prod': ReduceOp,
    'Pow': PowOp,
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
    'Sub': SubOp,
    'Sum': ReduceOp,
    'Square': SquareOp,
    'Squeeze': SqueezeOp,
    'StopGradient': StopGradientOp,
    'Switch': SwitchOp,
    'Tanh': TanhOp,
    'TanhGrad': TanhGradOp,
    'TensorArrayV3': TensorArrayOp,
    'Tile': TileOp,
    'Transpose': TransposeOp,
    'Unpack': UnpackOp,
    'UnsortedSegmentSum': UnsortedSegmentSumOp,
    'VariableV2': VariableOp,
    'Where': WhereOp,
    'ZerosLike': NumLikeOp,
}

TF_OP_TO_COUGR_REDUCE = {
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
# BatchMatMul
# L2Loss
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

TF_DTYPE_TO_COUGR = {
    tf.bool: DataType.bool,
    tf.int32: DataType.int32,
    tf.int64: DataType.int64,
    tf.uint8: DataType.uint8,
    tf.float32: DataType.float32,
    tf.string: DataType.string,
}

def tf_shape_to_cougr(tf_shape):
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
    cougr_graph = construct_cougr_graph(sess, sess.graph)
    # Clean up TF bits to avoid problems with successive graph loads
    sess.close()
    tf.reset_default_graph()
    return cougr_graph

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
    elif value_proto.dtype == types_pb2.DT_FLOAT:
        if tf_op.outputs[0].shape.ndims == 0 or \
           tf_op.outputs[0].shape.num_elements() == 1:
            value = value_proto.float_val
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
            assert list(value.shape) == tf_op.outputs[0].shape.as_list(), \
                'Op: {}, value: {}'.format(tf_op.name, value)
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

    # print(tf_op.op_def)
    # print(tf_op.node_def)

def construct_cougr_graph(tf_sess, tf_graph):
    graph = Graph()
    tensors = {}
    op_inputs = {}
    for tf_op in tf_graph._nodes_by_name.values():
        if tf_op.type in TF_OP_TO_COUGR.keys():
            # Map to CouGr op type
            cougr_type = TF_OP_TO_COUGR[tf_op.type]
        else:
            print('WARN: Unknown op type: {} (op: {})'
                  .format(tf_op.type, tf_op.name))
            cougr_type = UnknownOp

        # Create the CouGr internal op
        op = cougr_type(tf_op.name)

        if cougr_type == ReduceOp:
            reduce_op = None
            if tf_op.type in TF_OP_TO_COUGR_REDUCE:
                reduce_op = TF_OP_TO_COUGR_REDUCE[tf_op.type]
            else:
                print('WARN: Reduce may set reduction op: {}'.format(tf_op.type))
            if reduce_op is not None:
                op.setReductionOp(reduce_op)
            if tf_op.type == 'BiasAddGrad':
                op.setAxes(0)

        if cougr_type == ScatterUpdateOp:
            print('WARN: ScatterUpdate may set update op: {}'.format(tf_op.type))

        # Create the output tensors for this op
        for i in range(len(tf_op.outputs)):
            tf_tensor = tf_op.outputs[i]

            tf_dtype = tf_tensor.dtype.base_dtype
            if tf_dtype in TF_DTYPE_TO_COUGR.keys():
                cougr_dtype = TF_DTYPE_TO_COUGR[tf_dtype]
            else:
                print('WARN: Unknown dtype {} for tensor {}'
                      .format(tf_tensor.dtype, tf_tensor))
                cougr_dtype = None

            out_tens = Tensor(tf_tensor.name,
                tf_shape_to_cougr(tf_tensor.shape), cougr_dtype)
            tensors[out_tens.name] = out_tens
            op.addOutput(out_tens)

        # Track the input tensor names to connect them in next phase
        op_inputs[op.name] = []
        if tf_op.type == 'Split':
            # TF Split op has different interface than CouGr. Need to add the
            # size_splits tensor to match the CouGr interface (input[1])
            assert len(tf_op._inputs) == 2
            op_inputs[op.name].append(tf_op._inputs[1].name)
            # Signal to CouGr to use the num_split attribute by setting
            # size_splits equal to a scalar constant of value 0
            size_splits = cougr.constant('{}_size_splits'.format(op.name),
                                         out_shape=[], value=0, graph=graph)
            tensors[size_splits.name] = size_splits
            op_inputs[op.name].append(size_splits.name)
            op_inputs[op.name].append(tf_op._inputs[0].name)
        else:
            for i in range(len(tf_op.inputs)):
                op_inputs[op.name].append(tf_op.inputs[i].name)

        # Get the tf_op's attributes and set them as necessary
        parse_tf_op_attributes_into_op(tf_sess, tf_op, op)

        graph.addOp(op)

    # Hook up all the op inputs to the ops that generate them
    for op_name in op_inputs.keys():
        op = graph.opsByName[op_name]
        for in_tensor in op_inputs[op_name]:
            assert in_tensor in tensors.keys(), \
                   'Unknown input tensor {}'.format(in_tensor)
            graph.addInputToOp(op, tensors[in_tensor])

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
            # Get input ops and verify they are consts
            assert len(saver_op.inputs) == 3
            for in_tensor in saver_op.inputs:
                const_op = in_tensor.producer
                assert isinstance(const_op, ConstantOp)
                ops_to_remove.add(const_op)
            assert len(saver_op.outputs) >= 1
            # Restore ops can package all tensors together into a single
            # op, so need to traverse all outputs to their assign ops
            for out_tensor in saver_op.outputs:
                assert len(out_tensor.consumers) == 1
                for assign_op in out_tensor.consumers.values():
                    assert isinstance(assign_op, AssignOp)
                    ops_to_remove.add(assign_op)
        else:
            assert isinstance(saver_op, TFSaveOp)
            assert len(saver_op.inputs) >= 3
            for idx in range(3):
                in_tensor = saver_op.inputs[idx]
                const_op = in_tensor.producer
                assert isinstance(const_op, ConstantOp)
                ops_to_remove.add(const_op)
            assert len(saver_op.outputs) == 0
    model_name_consumers = []
    for model_name_op in model_name_ops:
        assert len(model_name_op.outputs) == 1
        model_name_consumers.extend(
            model_name_op.outputs[0].consumers.values())
    for saver_op in model_name_consumers:
        if saver_op not in ops_to_remove:
            # Only other op to catch is the control dependency op, which
            # is an IdentityOp and has no outputs
            assert isinstance(saver_op, IdentityOp)
            assert len(saver_op.outputs) == 1
            assert len(saver_op.outputs[0].consumers) == 0
            assert '/control_dependency' in saver_op.name
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
    for ctrl_op in control_ops:
        # Get all ops for the loop condition value calculation (1 and 2),
        # the variable contexts (3), and the loop body (4). Extract these
        # into a subgraph.
        subgraph_ops = [ctrl_op]
        visited_ops = set(subgraph_ops)
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
                'CouGr Framework(TF): Should be no up-stream control blocks!'
            visited_ops.add(next_op)
            if isinstance(next_op, EnterOp) or \
               isinstance(next_op, NextIterationOp):
                # Do not traverse past EnterOps, NextIterationOps
                continue
            if isinstance(next_op, MergeOp):
                frontier_ops.append(next_op)
            for in_tensor in next_op.inputs:
                bwd_frontier_ops.append(in_tensor.producer)

        # B) Traverse forward to get the SwitchOps, ExitOps, IdentityOps,
        #    body, NextIterationOps.
        fwd_frontier_ops = []
        for switch_op in frontier_ops:
            for out_tensor in switch_op.outputs:
                fwd_frontier_ops.extend(out_tensor.consumers.values())
        while len(fwd_frontier_ops) > 0:
            next_op = fwd_frontier_ops.pop(0)
            if next_op in visited_ops:
                continue
            if next_op.isControlOp():
                raise NotImplementedError(
                    'CouGr Framework(TF): Need nested control blocks')
            visited_ops.add(next_op)
            if isinstance(next_op, ExitOp):
                # Do not traverse past ExitOps
                continue
            for out_tensor in next_op.outputs:
                fwd_frontier_ops.extend(out_tensor.consumers.values())

        # [_] TODO (Joel): May need to go backward again to other EnterOps or to
        # identify the loop condition that gets executed...

        # Finally, create a ControlBlockOp (subgraph) with the main control
        # node as the ctrl_op, and add the ControlBlockOp to the CouGr graph
        # (which will move the graph ops into the subgraph)
        ctrl_block_op = ControlBlockOp('{}_block'.format(ctrl_op.name),
                                       ctrl_op, visited_ops)
        graph.addOp(ctrl_block_op)

    return graph
