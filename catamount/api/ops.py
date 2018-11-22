from ..graph import get_default_graph
from ..tensors import *
from ..ops.array_ops import *
from ..ops.ctrl_ops import *
from ..ops.constant import *
from ..ops.math_ops import *
from ..ops.placeholder import *
from ..ops.variable import *


def constant(name, out_shape, value=None, graph=None):
    if graph is None:
        graph = get_default_graph()

    const_op = ConstantOp(name)
    out_tensor = Tensor(name, TensorShape(out_shape))
    const_op.addOutput(out_tensor)
    graph.addOp(const_op)
    if value is not None:
        out_tensor.setValue(value)
    return out_tensor

def concat(name, out_shape, input_list, axis=0, graph=None):
    if graph is None:
        graph = get_default_graph()

    if not isinstance(axis, int):
        raise NotImplementedError(
            'catamount.concat axis yet unsupported type: {}'.format(type(axis)))

    concat_op = ConcatOp(name)
    out_tensor = Tensor(name, TensorShape(out_shape))
    concat_op.addOutput(out_tensor)
    graph.addOp(concat_op)
    for input in input_list:
        graph.addInputToOp(concat_op, input)
    # Finally, add the axis input tensor last (rank 0)
    axis_tensor = constant('{}:axis'.format(name), [], axis)
    graph.addInputToOp(concat_op, axis_tensor)
    return out_tensor

def dynamic_stitch(name, out_shape, indices_list=None, data_list=None,
                   graph=None):
    if graph is None:
        graph = get_default_graph()

    dynstitch_op = DynamicStitchOp(name)
    out_tensor = Tensor(name, TensorShape(out_shape))
    dynstitch_op.addOutput(out_tensor)
    graph.addOp(dynstitch_op)
    for input in indices_list:
        graph.addInputToOp(dynstitch_op, input)
    for input in data_list:
        graph.addInputToOp(dynstitch_op, input)
    return out_tensor

def enter(name, input, graph=None):
    if graph is None:
        graph = get_default_graph()

    enter_op = EnterOp(name)
    out_tensor = Tensor(name, TensorShape(input.shape))
    enter_op.addOutput(out_tensor)
    graph.addOp(enter_op)
    graph.addInputToOp(enter_op, input)
    return out_tensor

def expanddims(name, out_shape, input, axis=0, graph=None):
    if graph is None:
        graph = get_default_graph()

    if not isinstance(axis, int):
        raise NotImplementedError(
            'catamount.expanddims axis yet unsupported type: {}'
            .format(type(axis)))

    expanddims_op = ExpandDimsOp(name)
    out_tensor = Tensor(name, TensorShape(out_shape))
    expanddims_op.addOutput(out_tensor)
    graph.addOp(expanddims_op)
    graph.addInputToOp(expanddims_op, input)
    # Finally, add the axis input tensor last (rank 0)
    axis_tensor = constant('{}:axis'.format(name), [], axis)
    graph.addInputToOp(expanddims_op, axis_tensor)
    return out_tensor

def matmul(name, out_shape, in_a, in_b, graph=None):
    if graph is None:
        graph = get_default_graph()

    mm_op = MatMulOp(name)
    out_tensor = Tensor(name, TensorShape(out_shape))
    mm_op.addOutput(out_tensor)
    graph.addOp(mm_op)
    graph.addInputToOp(mm_op, in_a)
    graph.addInputToOp(mm_op, in_b)
    return out_tensor

def placeholder(name, out_shape, graph=None):
    if graph is None:
        graph = get_default_graph()

    ph_op = PlaceholderOp(name)
    out_tensor = Tensor(name, TensorShape(out_shape))
    ph_op.addOutput(out_tensor)
    graph.addOp(ph_op)
    return out_tensor

def pointwise(name, op_type, out_shape, in_a, in_b=None, graph=None):
    if graph is None:
        graph = get_default_graph()

    op = op_type(name)
    out_tensor = Tensor(name, TensorShape(out_shape))
    op.addOutput(out_tensor)
    graph.addOp(op)
    graph.addInputToOp(op, in_a)
    if in_b is not None:
        graph.addInputToOp(op, in_b)
    return out_tensor

def reduce(name, op_func, out_shape, input, axes=0, graph=None):
    if graph is None:
        graph = get_default_graph()

    op = ReduceOp(name, axes=axes)
    out_tensor = Tensor(name, TensorShape(out_shape))
    op.addOutput(out_tensor)
    graph.addOp(op)
    graph.addInputToOp(op, input)
    return out_tensor

def split(name, out_shape, input, size_splits=None, axis=0, num_split=2,
          graph=None):
    if graph is None:
        graph = get_default_graph()

    if size_splits is not None:
        raise NotImplementedError('Split needs to handle size_splits {}'
                                  .format(size_splits))

    # Instantiate op
    split_op = SplitOp(name)
    # Add num_split attribute
    if not isinstance(num_split, int):
        raise NotImplementedError('num_split of type {}'
                                  .format(type(num_split)))
    split_op.setNumSplit(num_split)
    # Add output tensors
    out_tensors = []
    for i in range(num_split):
        out_name = '{}_out{}'.format(name, i)
        out_tensors.append(Tensor(out_name, TensorShape(out_shape)))
        split_op.addOutput(out_tensors[i])
    graph.addOp(split_op)
    # Add inputs (tensor to split, size_splits, axis)
    graph.addInputToOp(split_op, input)
    if size_splits is None:
        # Pass scalar 0 as indicator that split should use num_split
        # attribute instead of size_splits
        size_splits_tensor = constant('{}_size_splits'.format(name),
                                      out_shape=[], value=0)
    else:
        assert isinstance(size_splits, Tensor)
        size_splits_tensor = size_splits
    graph.addInputToOp(split_op, size_splits_tensor)
    if isinstance(axis, int):
        axis_tensor = constant('{}_axis'.format(name), out_shape=[],
                               value=axis)
    else:
        assert isinstance(axis, Tensor)
        axis_tensor = axis
    graph.addInputToOp(split_op, axis_tensor)
    return out_tensors

def variable(name, out_shape, graph=None):
    if graph is None:
        graph = get_default_graph()

    var_op = VariableOp(name)
    out_tensor = Tensor(name, TensorShape(out_shape))
    var_op.addOutput(out_tensor)
    graph.addOp(var_op)
    return out_tensor

