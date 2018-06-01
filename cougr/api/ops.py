from ..graph import get_default_graph
from ..tensors import *
from ..ops.array_ops import *
from ..ops.math_ops import *
from ..ops.placeholder import *
from ..ops.variable import *


def concat(name, out_shape, input_list, axis=0, graph=None):
    if graph is None:
        graph = get_default_graph()

    concat_op = ConcatOp(name, axis=axis)
    out_tensor = Tensor(name, TensorShape(out_shape))
    concat_op.addOutput(out_tensor)
    graph.addOp(concat_op)
    for input in input_list:
        graph.addInputToOp(concat_op, input)
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

def split(name, out_shape, input, num_splits=2, axis=0, graph=None):
    if graph is None:
        graph = get_default_graph()

    split_op = SplitOp(name, num_splits=num_splits, axis=axis)
    out_tensors = []
    for i in range(num_splits):
        out_name = '{}_out{}'.format(name, i)
        out_tensors.append(Tensor(out_name, TensorShape(out_shape)))
        split_op.addOutput(out_tensors[i])
    graph.addOp(split_op)
    graph.addInputToOp(split_op, input)
    return out_tensors

def variable(name, out_shape, graph=None):
    if graph is None:
        graph = get_default_graph()

    var_op = VariableOp(name)
    out_tensor = Tensor(name, TensorShape(out_shape))
    var_op.addOutput(out_tensor)
    graph.addOp(var_op)
    return out_tensor

