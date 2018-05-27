import os
import tensorflow as tf

# To import graphs that use TF contrib libraries...
import tensorflow.contrib.mpi_collectives as mpi

from cougr.graph import *
from cougr.ops.array_ops import *
from cougr.ops.constant import *
from cougr.ops.init_ops import *
from cougr.ops.math_ops import *
from cougr.ops.placeholder import *
from cougr.ops.unknown_op import *
from cougr.ops.variable import *
from cougr.tensors.tensor import *

# Tools to import Tensorflow MetaGraphs into CouGr format

TF_OP_TO_COUGR = {
    'Add': AddOp,
    'Assign': AssignOp,
    'AssignAdd': AddOp, # Here, TF reuses the input tensor for output
    'AssignSub': SubOp, # Here, TF reuses the input tensor for output
    'BiasAdd': AddOp, # Here, TF special-case for 1D bias input
    'Cast': CastOp,
    'ConcatV2': ConcatOp,
    'Const': ConstOp,
    'Conv2D': Conv2DOp,
    'Exp': ExpOp,
    'FloorDiv': BasePointwiseOp,
    'FloorMod': BasePointwiseOp,
    'Identity': IdentityOp,
    'Less': LessOp,
    'LogicalNot': LogicalNotOp,
    'MatMul': MatMulOp,
    'Maximum': MaximumOp,
    'Mean': ReduceOp,
    'Minimum': MinimumOp,
    # tf.contrib.mpi_collectives.MPIInit has no compute graph function
    'MPIInit': NoOp,
    # tf.contrib.mpi_collectives.MPISize behaves like a placeholder
    'MPISize': PlaceholderOp,
    'Mul': MulOp,
    'Neg': NegOp,
    'NoOp': NoOp, # Ignore no-ops
    'NotEqual': NotEqualOp,
    'Pack': StackOp,
    'Placeholder': PlaceholderOp,
    'Prod': ReduceOp,
    'Pow': PowOp,
    'RandomUniform': RandomInitializerOp,
    'RealDiv': BasePointwiseOp,
    'Relu': ReluOp,
    'Reduce': ReduceOp,
    'Reshape': ReshapeOp,
    'RestoreV2': NoOp, # Ignore Restore ops
    'Rsqrt': RsqrtOp,
    'SaveV2': NoOp, # Ignore Saver ops
    'Shape': ShapeOp,
    'Sigmoid': SigmoidOp,
    'SplitV': SplitOp,
    'Sqrt': SqrtOp,
    'StridedSlice': StridedSliceOp,
    'Sub': SubOp,
    'Sum': ReduceOp,
    'Tanh': TanhOp,
    'VariableV2': VariableOp,
}

# [_] TODO (Joel): Add these for ResNets!
# AddN
# ApplyMomentum
# BiasAddGrad
# BroadcastGradientArgs
# Conv2DBackpropFilter
# Conv2DBackpropInput
# DynamicStitch
# ExpandDims
# FIFOQueueV2
# Fill
# FusedBatchNorm
# FusedBatchNormGrad
# InTopK
# MaxPool
# MaxPoolGrad
# MergeSummary
# PreventGradient
# QueueCloseV2
# QueueDequeueV2
# QueueEnqueueV2
# QueueSizeV2
# Range
# ReluGrad
# ScalarSummary
# SparseSoftmaxCrossEntropyWithLogits
# Stage
# Tile
# TruncatedNormal
# Unstage
# ZerosLike


TF_DTYPE_TO_COUGR = {
    tf.bool: DataType.bool,
    tf.int32: DataType.int32,
    tf.int64: DataType.int64,
    tf.uint8: DataType.uint32,
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

def import_graph(tf_filename):
    if '.meta' not in tf_filename or not os.path.exists(tf_filename):
        raise FileNotFoundError('ERROR: Invalid file {}. Must be .meta file'
            .format(tf_filename))
    saver = tf.train.import_meta_graph(tf_filename)
    sess = tf.Session()
    try:
        tf_model_name = tf_filename.replace('.meta', '')
        saver.restore(sess, tf_model_name)
    except Exception:
        print('WARN: Cannot find checkpoint data {}, trying to proceed'
              .format('{}.data-?-of-?'.format(tf_filename)))

    tf_graph = sess.graph
    graph = Graph()
    tensors = {}
    op_inputs = {}
    for tf_op_name in tf_graph._nodes_by_name.keys():
        tf_op = tf_graph._nodes_by_name[tf_op_name]
        if tf_op.type in TF_OP_TO_COUGR.keys():
            # Map to CouGr op type
            cougr_type = TF_OP_TO_COUGR[tf_op.type]
        else:
            print('WARN: Unknown op type: {}'.format(tf_op.type))
            cougr_type = UnknownOp

        # Create the CouGr internal op
        op = cougr_type(tf_op_name)

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
        for i in range(len(tf_op.inputs)):
            op_inputs[op.name].append(tf_op.inputs[i].name)

        graph.addOp(op)

#    print('Ops: {}'.format(graph.opsByName))
#    print('Tensors: {}'.format(tensors))

    # Hook up all the op inputs to the ops that generate them
    for op_name in op_inputs.keys():
        op = graph.getOpByName(op_name)
        for in_tensor in op_inputs[op_name]:
            assert in_tensor in tensors.keys(), \
                   'Unknown input tensor {}'.format(in_tensor)
            graph.addInputToOp(op, tensors[in_tensor])

    return graph
