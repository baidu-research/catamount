import numpy as np
import sympy
from .base_op import Op
from ..api import utils
from ..tensors.tensor import DataType
from ..tensors.tensor_shape import Dimension, TensorShape


class BroadcastGradientArgsOp(Op):
    ''' Returns the indices of dimensions that were reduced for computing
        the forward propagated op, OP(input[0], input[1]), where OP supports
        broadcasting. This is used for calculating the dimensions of a
        gradient tensor to broadcast into the corresponding backprop to OP.
        First output (0) represents the dimensions of the first input tensor
        to OP that must be broadcast for backprop. Second output is analogous
    '''
    def __init__(self, name):
        super(BroadcastGradientArgsOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(len(self._inputs) == 2)
        self.debugAssert(len(self._outputs) == 2)

        if self._inputs[0].value is None or \
           self._inputs[1].value is None:
            # Cannot calculate outputs, so return
            return

        in_0_val = np.copy(self._inputs[0].value)
        in_1_val = np.copy(self._inputs[1].value)
        out_0_val = []
        out_1_val = []
        while len(in_0_val) < len(in_1_val):
            in_0_val = np.concatenate(([1], in_0_val))
        while len(in_0_val) > len(in_1_val):
            in_1_val = np.concatenate(([1], in_1_val))
        for idx in range(len(in_0_val)):
            if in_0_val[idx] == 1:
                out_0_val.append(idx)
            elif in_1_val[idx] == 1:
                out_1_val.append(idx)
            else:
                if in_0_val[idx] - in_1_val[idx] != 0:
                    print('WARN: BroadcastGradientArgs op {}: Dimension[{}]' \
                          ' mismatch: {} {}'.format(self.name, idx,
                          in_0_val[idx], in_1_val[idx]))
        self._outputs[0].mergeShape([len(out_0_val)],
                                    make_symbolic=make_symbolic)
        self._outputs[0].setValue(out_0_val)
        self._outputs[1].mergeShape([len(out_1_val)],
                                    make_symbolic=make_symbolic)
        self._outputs[1].setValue(out_1_val)

    def calcAlgFlops(self):
        # Only handles shapes, so no Flops
        return 0

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        return self.bytesAccessOutput()


class ConcatOp(Op):
    def __init__(self, name):
        super(ConcatOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        # Verify input shapes can be merged when concat axis is masked
        self.debugAssert(len(self._inputs) >= 2)
        self.debugAssert(self._inputs[-1].shape.rank == 0)
        axis = self._inputs[-1].value
        self.debugAssert(axis is not None,
                         'Op {} axis is None'.format(self._name))

        self.debugAssert(len(self._outputs) == 1)
        out_shape_c_dim = Dimension(0)
        propagate_values = True
        for idx in range(len(self._inputs) - 1):
            in_tensor = self._inputs[idx]
            if in_tensor.value is None:
                propagate_values = False
            in_shape = TensorShape(in_tensor.shape.dims)
            in_c_dim = in_shape.dims[axis]
            out_shape_c_dim += in_c_dim
            in_shape.dims[axis] = Dimension(None)
            self._outputs[0].mergeShape(in_shape,
                                        make_symbolic=make_symbolic)
        self._outputs[0].shape.setDimension(axis, out_shape_c_dim,
                                            make_symbolic=make_symbolic)

        if not propagate_values:
            return

        # If the input values are fully specified, then also propagate
        # those values to the outputs
        out_value = None
        for idx in range(len(self._inputs) - 1):
            if out_value is None:
                out_value = self._inputs[idx].value
            else:
                out_value = np.concatenate(
                    (out_value, self._inputs[idx].value), axis=axis)
        self._outputs[0].setValue(out_value)
        self.debugAssert(self._inputs[0].isValid())

    def calcAlgFlops(self):
        # ConcatOps have no Flops
        return 0

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class ConcatOffsetOp(Op):
    ''' Calculate the offsets within the output of a concat operation for
        its backpropagation. The first input (0) is the axis for the
        concat operation, and the rest of the inputs are offsets of the
        input tensors within the output of the concat.
    '''
    def __init__(self, name):
        super(ConcatOffsetOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(len(self._inputs) - 1 == len(self._outputs))
        # Find an input with resolved shape
        in_shape = None
        inputs_rank = None
        concat_axis = self._inputs[0].value
        for idx in range(1, len(self._inputs)):
            if self._inputs[idx].value is not None:
                in_shape = np.zeros(self._inputs[idx].value.shape)
                inputs_rank = len(self._inputs[idx].value)
                break
        if inputs_rank is None or concat_axis is None:
            # Unable to resolve output dimensions from inputs
            return
        curr_axis_offset = 0
        for idx in range(len(self._outputs)):
            self._outputs[idx].mergeShape([inputs_rank],
                                          make_symbolic=make_symbolic)
            curr_in_shape = self._inputs[idx + 1].value
            if curr_in_shape is None:
                self.notImplemented('ConcatOffset: Continue resolving?')
                break
            out_shape = in_shape.tolist()
            out_shape[concat_axis] = curr_axis_offset
            curr_axis_offset += curr_in_shape[concat_axis]
            self._outputs[idx].setValue(out_shape)

    def calcAlgFlops(self):
        # Calculation of concat offset is not algorithmic
        return 0

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensors, which must be accessed
        return self.bytesAccessOutput()


class DynamicStitchOp(Op):
    def __init__(self, name):
        super(DynamicStitchOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(len(self._inputs) % 2 == 0)
        self.debugAssert(len(self._outputs) == 1)
        # NOTE: If not all input index tensors are fully specified, then
        # we cannot infer the output shape and must pass. If not all data
        # tensors are fully specified, we cannot calculate the output value
        can_prop_shape = True
        can_prop_values = True
        index_arrs = []
        data_arrs = []
        extra_rank = -1
        extra_dims = []
        data_values_list = []
        for idx, in_tensor in enumerate(self._inputs):
            if idx < len(self._inputs) // 2:
                if in_tensor.value is None:
                    can_prop_shape = False
                index_arrs.append(np.array(self._inputs[idx].value))
            else:
                curr_data_idx = len(data_arrs)
                data_arr = np.array(self._inputs[idx].value)
                data_arrs.append(data_arr)
                if self._inputs[idx].value is None:
                    can_prop_values = False
                    continue
                # Assert that the index and data shapes are the same
                index_arr = index_arrs[curr_data_idx]
                data_arr = data_arrs[curr_data_idx]
                this_extra_rank = data_arr.ndim - index_arr.ndim
                self.debugAssert(this_extra_rank >= 0)
                if extra_rank == -1:
                    extra_rank = this_extra_rank
                    dims = self._inputs[idx].shape.dims
                    extra_dims = dims[len(dims) - extra_rank:]
                else:
                    self.debugAssert(extra_rank == this_extra_rank)
                    dims = self._inputs[idx].shape.dims
                    this_extra_dims = dims[len(dims) - extra_rank:]
                    for e_dim, te_dim in zip(extra_dims, this_extra_dims):
                        self.debugAssert(e_dim == te_dim)
        self.debugAssert(len(index_arrs) == len(data_arrs))
        if can_prop_shape:
            # Find the max index
            max_index = -1
            for ind_arr in index_arrs:
                for idx in ind_arr.flatten():
                    max_index = max(idx, max_index)
            self.debugAssert(isinstance(max_index, sympy.Expr) or
                             max_index >= 0)
            out_shape = [max_index + 1]
            for rank_id in range(extra_rank):
                out_shape.append(Dimension(extra_dims[rank_id]))
            self._outputs[0].mergeShape(out_shape,
                                        make_symbolic=make_symbolic)

        if can_prop_values:
            # Migrate values from data_arrs to output
            out_values = []
            for inds, data in zip(index_arrs, data_arrs):
                for loc, idx in np.ndenumerate(inds):
                    while idx >= len(out_values):
                        out_values.append(None)
                    out_values[idx] = data[loc]
            self._outputs[0].setValue(out_values)

    def calcAlgFlops(self):
        # DynamicStitch only merges array, so no Flops
        return 0

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class ExpandDimsOp(Op):
    def __init__(self, name):
        super(ExpandDimsOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(len(self._inputs) == 2)
        self.debugAssert(len(self._outputs) == 1)
        axis = self._inputs[1].value
        if axis is None:
            print('WARN: Undetermined axis for ExpandDimsOp {}'
                  .format(self._name))
            return
        elif isinstance(axis, (list, np.ndarray)):
            self.debugAssert(len(axis) == 1)
            axis = axis[0]
        if self._inputs[0].shape.rank == 0:
            self.debugAssert(axis == 0)
            if not self._outputs[0].shape.isFullyNumeric():
                self.notImplemented(
                    'ExpandDimsOp propagateShapes scalar output')
            out_value = [self._inputs[0].value]
            self._outputs[0].setValue(out_value)
        else:
            out_shape = list(self._inputs[0].shape.dims)
            self.debugAssert(axis <= len(out_shape) and \
                             axis >= (-len(out_shape) - 1))
            if axis < 0:
                axis += len(out_shape) + 1
            out_shape.insert(axis, 1)
            self._outputs[0].mergeShape(out_shape,
                                        make_symbolic=make_symbolic)
            if self._inputs[0].value is not None:
                self.notImplemented('ExpandDimsOp propagate vals rank 1+')

    def calcAlgFlops(self):
        # ExpandDimsOp has no algorithmic Flops
        return 0

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class FillOp(Op):
    ''' Returns a tensor of the shape specified in first input tensor (and
        fills that tenros with a set value passed in second input).
        in each location of the output tensor. Similar to Numpy and TF
        Fill.
    '''
    def __init__(self, name):
        super(FillOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        # FillOps should create an output tensor with the shape passed
        # as the first input tensor
        self.debugAssert(len(self._inputs) == 2, 'FillOp {} has {} inputs' \
            .format(self._name, len(self._inputs)))
        self.debugAssert(self._inputs[1].shape.isScalar())
        self.debugAssert(len(self._outputs) == 1, 'FillOp {} has {} outputs' \
            .format(self._name, len(self._outputs)))
        # The first input tensor contains the shape of output tensor
        out_shape = self._inputs[0].value
        if out_shape is None:
            print('WARN: FillOp {} shape undetermined'.format(self._name))
            return
        self._outputs[0].mergeShape(out_shape,
                                    make_symbolic=make_symbolic)
        if self._outputs[0].shape.isFullyNumeric():
            if self._outputs[0].shape.isScalar():
                self._outputs[0].setValue(self._inputs[1].value)
            else:
                out_shape = self._outputs[0].shape.asList()
                elt_value = self._inputs[1].value
                value = np.full(out_shape, elt_value)
                self._outputs[0].setValue(value)

    def calcAlgFlops(self):
        # FillOps have no Flops
        return 0

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class GatherOp(Op):
    def __init__(self, name):
        super(GatherOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        # First input is the tensor from which to gather. Second input is
        # the indices to gather from first tensor.
        self.debugAssert(len(self._inputs) == 2 or \
                         len(self._inputs) == 3)
        self.debugAssert(len(self._outputs) == 1)
        axis = 0 # TODO (Joel): Optional third tensor can specify axis
        if len(self._inputs) == 3:
            axis = int(self._inputs[2].value)
        if axis == 0:
            out_dims = []
            for dim_idx in range(self._inputs[1].shape.rank):
                out_dims.append(self._inputs[1].shape.getDimension(dim_idx))
            for dim_idx in range(1, self._inputs[0].shape.rank):
                out_dims.append(self._inputs[0].shape.getDimension(dim_idx))
            self._outputs[0].mergeShape(out_dims,
                                        make_symbolic=make_symbolic)
        else:
           self.notImplemented('GatherOp propagateShapes: Non-zero axis')

        if self._inputs[0].value is not None and \
           self._inputs[1].value is not None and \
           (len(self._inputs) == 2 or self._inputs[2].value is not None):
            axis = 0
            if len(self._inputs) == 3:
                axis = int(self._inputs[2].value)
            out_val = np.take(self._inputs[0].value, self._inputs[1].value,
                              axis=axis)
            self._outputs[0].setValue(out_val)

    def calcAlgFlops(self):
        # GatherOps have no Flops
        return 0

    def calcAlgBytes(self):
        # NOTE: The data in the output is read from input[0], but the op
        # does not read all of input[0].
        return self._inputs[1].size + 2 * self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class InvertPermutationOp(Op):
    def __init__(self, name):
        super(InvertPermutationOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(len(self._inputs) == 1)
        self.debugAssert(self._inputs[0].dtype == DataType.int32)
        self.debugAssert(len(self._outputs) == 1)
        self.debugAssert(self._outputs[0].dtype == DataType.int32)
        if self._inputs[0].shape.isUnknown():
            # Cannot propagate unknown shape
            return
        self._outputs[0].mergeShape(self._inputs[0].shape,
                                    make_symbolic=make_symbolic)

        if self._inputs[0].value is None:
            # Cannot propagate unknown value
            return

        out_value = self._inputs[0].value
        self.debugAssert(isinstance(out_value, np.ndarray))
        out_value = np.argsort(out_value)
        self._outputs[0].setValue(out_value)

    def calcAlgFlops(self):
        # InvertPermutationOps have no Flops
        return 0

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class ListDiffOp(Op):
    def __init__(self, name):
        super(ListDiffOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(len(self._inputs) == 2)
        self.debugAssert(self._inputs[0].shape.rank == 1)
        self.debugAssert(self._inputs[1].shape.rank == 1)
        self.debugAssert(len(self._outputs) == 2)
        if self._inputs[0].shape.isUnknown() or \
           self._inputs[1].shape.isUnknown() or \
           self._inputs[0].value is None or \
           self._inputs[1].value is None:
            # Can only resolve shapes and outputs if input values are
            # fully specified
            return
        in_0_val = np.array(self._inputs[0].value)
        in_1_val = np.array(self._inputs[1].value)
        out_val = np.setdiff1d(in_0_val, in_1_val)
        out_shape = [len(out_val)]
        self._outputs[0].mergeShape(out_shape,
                                    make_symbolic=make_symbolic)
        self._outputs[0].setValue(out_val)
        indices = []
        for idx in range(len(in_0_val)):
            if in_0_val[idx] in out_val:
                indices.append(idx)
        self.debugAssert(len(indices) == len(out_val))
        self._outputs[1].mergeShape(out_shape,
                                    make_symbolic=make_symbolic)
        self._outputs[1].setValue(indices)

    def calcAlgFlops(self):
        # ListDiff requires look-ups in a set and comparisons, but it may
        # be tough to count comparisons. This number should be relatively
        # small, so assume 0 Flops
        return 0

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        return self.bytesAccessOutput()


class OneHotOp(Op):
    def __init__(self, name):
        super(OneHotOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(len(self._inputs) == 4)
        self.debugAssert(len(self._outputs) == 1)

        # First input is samples to be one-hot encoded, so first dimensions
        # of output must be same as first input dimensions
        if self._inputs[0].shape.isUnknown():
            return
        out_shape = list(self._inputs[0].shape.dims)
        # Second input is the length of the new dimension
        if self._inputs[1].shape.isUnknown() or self._inputs[1].value == None:
            out_shape.append(None)
        else:
            self.debugAssert(isinstance(self._inputs[1].value,
                                        (int, sympy.Expr)))
            out_shape.append(self._inputs[1].value)
        self._outputs[0].mergeShape(out_shape,
                                    make_symbolic=make_symbolic)

    def calcAlgFlops(self):
        # One-hot representations have no Flops
        return 0

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class NumLikeOp(Op):
    ''' Returns a tensor of the same shape as input with a set value
        in each location of the output tensor. Similar to Numpy and TF
        ZerosLike and OnesLike.
    '''
    def __init__(self, name):
        super(NumLikeOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        # NumLikeOps should create an output tensor with the shape passed
        # as the first input tensor
        self.debugAssert(len(self._inputs) == 1, 'NumLikeOp {} has {} inputs' \
            .format(self._name, len(self._inputs)))
        self.debugAssert(len(self._outputs) == 1, 'NumLikeOp {} has {} outputs' \
            .format(self._name, len(self._outputs)))
        # The output tensor should be the same shape as the input
        if not self._inputs[0].shape.isUnknown():
            if self._inputs[0].shape != self._outputs[0].shape:
                self.notImplemented('NumLikeOp propagateShapes {}'
                                    .format(self._name))
            self._outputs[0].mergeShape(self._inputs[0].shape,
                                        make_symbolic=make_symbolic)

    def calcAlgFlops(self):
        # NumLikeOps have no Flops
        return 0

    def calcAlgBytes(self):
        # Only need to read the shape of the input to write the output
        return self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class PadOp(Op):
    def __init__(self, name):
        super(PadOp, self).__init__(name)
        # TODO: Need to set the configuration of the padding. Paddings can be
        # constants, reflections across axes, or symmetric (etc.?).
        # TODO: Write a test for this op

    def propagateShapes(self, make_symbolic=False):
        # Input 0 is the tensor to be padded, and input 1 is the padding
        # configuration. The padding configuration is dimension [n, 2], where
        # n is the rank of input[0], and the second dimension is how many
        # elements to add in that dimension above and below the data.
        self.debugAssert(len(self._inputs) == 2)
        self.debugAssert(len(self._outputs) == 1)

        in_shape = self._inputs[0].shape
        out_shape = []
        pad_cfg = self._inputs[1].value
        self.debugAssert(in_shape.rank == len(pad_cfg))
        for idx, dim in enumerate(in_shape.dims):
            new_dim = Dimension(dim)
            new_dim += pad_cfg[idx][0]
            new_dim += pad_cfg[idx][1]
            out_shape.append(new_dim)
        self._outputs[0].mergeShape(out_shape, make_symbolic=make_symbolic)

        # TODO: Propagate values if desired

    def calcAlgFlops(self):
        # All handling here is for memory accesses to the tensors
        return 0

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class RankOp(Op):
    def __init__(self, name):
        super(RankOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(len(self._inputs) == 1)
        self.debugAssert(len(self._outputs) == 1)
        # Output must be a scalar (no need for make_symbolic, since scalar)
        self._outputs[0].mergeShape([])
        if not self._inputs[0].shape.isUnknown():
            self._outputs[0].setValue(self._inputs[0].shape.rank)

    def calcAlgFlops(self):
        # RankOps have no Flops
        return 0

    def calcAlgBytes(self):
        # Rank only write the rank to the output tensor, but does not read input
        return self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class ReshapeOp(Op):
    def __init__(self, name):
        super(ReshapeOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        # The first input is reshaped according to the second tensor
        # 1) If the second tensor is [], try to reshape the input to a scalar
        # 2) If the second tensor contains -1, then reshape according to the
        #    other dimensions of second tensor and fill the -1 dimension to
        #    maintain the same number of elements.
        self.debugAssert(len(self._inputs) == 2)
        self.debugAssert(len(self._outputs) == 1)

        num_elts = self._inputs[0].shape.numElements()
        if self._inputs[1].shape.isScalar():
            self._outputs[0].mergeShape([num_elts],
                                        make_symbolic=make_symbolic)
        else:
            if self._inputs[1].value is None:
                if self._inputs[1].shape.isUnknown():
                    self.notImplemented('Reshape {} in1 val {} shape {}'
                        .format(self.name, self._inputs[1].value,
                                self._inputs[1].shape))
                else:
                    self.debugAssert(self._inputs[1].shape.rank == 1)
                    if self._inputs[1].shape.dims[0].value == 1:
                        # Assume that the output must be the flattened
                        # input[0] values, so the shape is the same as the
                        # number of input[0] values.
                        self._outputs[0].mergeShape([num_elts],
                                                    make_symbolic=make_symbolic)
                    elif self._inputs[0].shape.rank == \
                         self._inputs[1].shape.dims[0].value:
                        # Try merging input and output shapes
                        self._outputs[0].mergeShape(self._inputs[0].shape,
                            make_symbolic=make_symbolic)
                    else:
                        # TODO: Cannot resolve shapes (unknown input[1] value)
                        print('WARN: Reshape {} impl: in1 val {} shape {}'
                              .format(self.name, self._inputs[1].value,
                                      self._inputs[1].shape))
            else:
                out_shape = self._inputs[1].value
                self.debugAssert(isinstance(out_shape, np.ndarray))
                out_shape = out_shape.tolist()
                prod_dims = None
                minus_1_idx = None
                for idx, val in enumerate(out_shape):
                    if val == -1:
                        self.debugAssert(minus_1_idx is None)
                        minus_1_idx = idx
                    else:
                        self.debugAssert(isinstance(val, (int, sympy.Expr,
                                                          np.int64)))
                        if prod_dims is None:
                            prod_dims = 1
                        prod_dims *= val
                if minus_1_idx is not None:
                    if prod_dims is None:
                        minus_1_dim = num_elts
                    else:
                        self.debugAssert(num_elts % prod_dims == 0,
                            'Num elements {} not divisible by removed dims ' \
                            'product {}'.format(num_elts, prod_dims))
                        minus_1_dim = num_elts // prod_dims
                    out_shape[minus_1_idx] = minus_1_dim
                else:
                    if prod_dims != num_elts:
                        print('WARN: Reshape {} implement condition: {} {}'
                              .format(self.name, prod_dims, num_elts))
                        return
                    self.debugAssert(prod_dims == num_elts)
                self._outputs[0].mergeShape(out_shape,
                                            make_symbolic=make_symbolic)

        if self._outputs[0].shape.isFullyNumeric() and \
           self._inputs[0].value is not None:
            self._outputs[0].setValue(
                np.reshape(self._inputs[0].value,
                           self._outputs[0].shape.asList()))

    def calcAlgFlops(self):
        # ReshapeOps have no Flops
        return 0

    def calcAlgBytes(self):
        # WARNING: Depending on the memory layout of the input and whether
        # the op can forward tensor references rather than copying, this op
        # may only need to read and write the shape, rather than copying the
        # whole tensor. Assume worst-case for now
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class ReverseSequenceOp(Op):
    def __init__(self, name):
        super(ReverseSequenceOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(len(self._inputs) >= 1)
        self.debugAssert(len(self._outputs) == 1)
        # For now, assume reversing sequence dimension does not change
        # the size from input to output tensor.
        self._outputs[0].mergeShape(self._inputs[0].shape,
                                    make_symbolic=make_symbolic)

    def calcAlgFlops(self):
        # Reversing a sequence does not require Flops
        return 0

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class ScatterOp(Op):
    def __init__(self, name):
        super(ScatterOp, self).__init__(name)

    def calcAlgFlops(self):
        # ScatterOps have no Flops
        return 0

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()


class ShapeOp(Op):
    def __init__(self, name):
        super(ShapeOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        # There should be the same number of outputs as inputs, and
        # all outputs should be 1D tensors with specified dimension
        # equal to the rank of the corresponding input tensor
        self.debugAssert(len(self._inputs) == len(self._outputs))
        for idx in range(len(self._inputs)):
            out_dim_0 = self._inputs[idx].shape.rank
            # Currently, rank of a tensor is always defined, so do not allow
            # the output shape to be set to a symbol
            self._outputs[idx].mergeShape([out_dim_0],
                                          make_symbolic=False)

        # Can set some output values if the input shape is known
        for idx in range(len(self._inputs)):
            if not self._inputs[idx].shape.isUnknown():
                out_value = []
                for dim_idx in range(self._inputs[idx].shape.rank):
                    # Can propagate symbols and values, so request symbols
                    dim = self._inputs[idx].shape.getDimension(dim_idx)
                    # Prefer to propagate symbols before values, because
                    # symbols can be resolved back to values, but not
                    # vice versa
                    if dim._symbol is not None:
                        out_val = dim._symbol
                    else:
                        out_val = dim._value
                    out_value.append(out_val)
                self._outputs[idx].setValue(out_value)

    def calcAlgFlops(self):
        # ShapeOps have no Flops
        return 0

    def calcAlgBytes(self):
        # Only need to write out the shape of the tensor
        return self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class SliceOp(Op):
    def __init__(self, name):
        super(SliceOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(len(self._inputs) == 3)
        self.debugAssert(len(self._outputs) == 1)

        if not self._inputs[0].shape.isFullySymbolic() or \
           self._inputs[1].value is None or self._inputs[2].value is None:
            # Cannot resolve output dimensions, so return
            return

        # Output shape must be same rank as first input, and must be
        # calculated from second and third inputs
        in_shape = self._inputs[0].shape
        begin_vals = self._inputs[1].value
        size_vals = self._inputs[2].value
        end_vals = []
        out_shape = []
        for idx, dim in enumerate(in_shape.dims):
            self.debugAssert(dim.symbol is not None)
            begin_val = begin_vals[idx]
            size_val = size_vals[idx]
            if size_val == -1:
                size_val = dim.symbol - begin_val
            end_val = begin_val + size_val
            if not isinstance(end_val, sympy.Expr) and \
               not isinstance(dim.symbol, sympy.Expr):
                self.debugAssert(end_val <= dim.symbol)
            end_vals.append(end_val)
            out_shape.append(size_val)
        self._outputs[0].mergeShape(out_shape,
                                    make_symbolic=make_symbolic)

        # Finally, if possible, propagate values
        if self._inputs[0].value is not None and \
           self._inputs[1].value is not None and \
           self._inputs[2].value is not None:
            if len(self._inputs[1].value) > 1 or \
               len(self._inputs[2].value) > 1:
                self.notImplemented('SliceOp propagate values rank 2+')
            out_value = self._inputs[0].value[begin_vals[0]:end_vals[0]]
            self._outputs[0].setValue(out_value)

    def calcAlgFlops(self):
        # SliceOps have no Flops
        return 0

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class SizeOp(Op):
    def __init__(self, name):
        super(SizeOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(len(self._inputs) == 1)
        self.debugAssert(len(self._outputs) == 1)
        if self._outputs[0].shape.isUnknown():
            self._outputs[0].mergeShape([], make_symbolic=make_symbolic)
        else:
            self.debugAssert(self._outputs[0].shape.isScalar())
        self._outputs[0].setValue(self._inputs[0].shape.numElements())

    def calcAlgFlops(self):
        # SizeOps have no Flops
        return 0

    def calcAlgBytes(self):
        # Only need to write out the size of the tensor
        return self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class SplitOp(Op):
    ''' Split the input tensor (input[0]) either using the shape supplied as
        the second input (input[1]) or if the second input is _____TODO______,
        then split evenly according to num_split attribute. Split along axis
        specified in third input (input[2]) or the axis attribute (TODO).
    '''
    def __init__(self, name):
        super(SplitOp, self).__init__(name)
        self._num_split = None

    def setNumSplit(self, num_split):
        self._num_split = num_split

    def propagateShapes(self, make_symbolic=False):
        # input[0] is the tensor to split
        # input[1] is the sizes of the split or a scalar 0 if using num_split
        # input[2] is the axis of the split
        self.debugAssert(len(self._inputs) == 3)
        self.debugAssert(self._num_split is None or \
                         len(self._outputs) == self._num_split)

        # Check size_splits tensor first
        # Cases to fall back on num_split:
        # 1) if size_split is a scalar of value 0
        # 2) if size_split value is None
        # TODO (Joel): Finish implementation when size_splits is specified!
        size_splits = self._inputs[1]
        axis = self._inputs[2].value
        self.debugAssert(axis is not None)
        in_tensor = self._inputs[0]
        if size_splits.shape.isScalar() and size_splits.value != 0:
            self.notImplemented('Handle non-zero size_splits')

        elif not size_splits.shape.isScalar() and \
             size_splits.value is not None:
            total_size = np.add.reduce(size_splits.value)
            in_tensor_axis_size = in_tensor.shape.getDimension(axis).value
            self.debugAssert(total_size == in_tensor_axis_size)
            out_dims = list(in_tensor.shape.asList())
            for idx, axis_val in enumerate(size_splits.value):
                out_dims[axis] = axis_val
                self._outputs[idx].mergeShape(out_dims,
                                              make_symbolic=make_symbolic)
        else:
            # Case where op should split evenly according to num_splits
            # as indicated by size_splits == 0
            self.debugAssert(size_splits.value == 0)
            if self._num_split is None:
                num_split = len(self._outputs)
            else:
                num_split = self._num_split

            out_shape = TensorShape(in_tensor.shape)
            self.debugAssert(axis < len(out_shape.dims))
            out_dim = out_shape.dims[axis]
            if out_dim.value is not None:
                self.debugAssert(out_dim._value % num_split == 0)
                out_dim._value //= num_split
            if out_dim._symbol is not None:
                out_dim._symbol = -(-out_dim._symbol // num_split)
            out_shape.dims[axis] = out_dim
            for out_tensor in self.outputs:
                out_tensor.mergeShape(out_shape,
                                      make_symbolic=make_symbolic)

        # TODO (Joel): Can split if input is specified

    def calcAlgFlops(self):
        # SplitOps have no Flops
        return 0

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class SqueezeOp(Op):
    ''' SqueezeOps remove dimensions of size 1 of a tensor. Since this
    operation does not change the ordering of dimension, the memory layout
    of the input should be propagated to the output.
    '''
    def __init__(self, name):
        super(SqueezeOp, self).__init__(name)
        self._squeeze_dims = []

    def setSqueezeDims(self, squeeze_dims=None):
        if squeeze_dims is not None:
            self._squeeze_dims.extend(squeeze_dims)

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(len(self._inputs) > 0)
        self.debugAssert(len(self._outputs) == 1)
        if len(self._inputs) == 1:
            in_shape = self._inputs[0].shape.dims
            out_shape = []
            for idx, dim in enumerate(in_shape):
                if dim.value == 1 and ((len(self._squeeze_dims) == 0) or \
                                       idx in self._squeeze_dims):
                    # Remove all dimensions == 1 except those *not* included
                    # in (unempty) self._squeeze_dims
                    pass
                else:
                    out_shape.append(dim)
            self._outputs[0].mergeShape(out_shape,
                                        make_symbolic=make_symbolic)
        else:
            self.notImplemented('Squeeze propagateShapes multi-input')

    def calcAlgFlops(self):
        # SqueezeOps have no Flops
        return 0

    def calcAlgBytes(self):
        # WARNING: Currently, we assume that the tensor must be copied
        # from input to output to ensure that it does not get overwritten
        # by other operations.
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class StridedSliceOp(Op):
    def __init__(self, name):
        super(StridedSliceOp, self).__init__(name)
        self._begin_mask = 0
        self._ellipsis_mask = 0
        self._end_mask = 0
        self._new_axis_mask = 0
        self._shrink_axis_mask = 0

    def setBeginMask(self, mask):
        self.debugAssert(isinstance(mask, int))
        self._begin_mask = mask

    def setEllipsisMask(self, mask):
        self.debugAssert(isinstance(mask, int))
        self._ellipsis_mask = mask

    def setEndMask(self, mask):
        self.debugAssert(isinstance(mask, int))
        self._end_mask = mask

    def setNewAxisMask(self, mask):
        self.debugAssert(isinstance(mask, int))
        self._new_axis_mask = mask

    def setShrinkAxisMask(self, mask):
        self.debugAssert(isinstance(mask, int))
        self._shrink_axis_mask = mask

    def isIndexSet(self, mask, bit_idx):
        return ((mask >> bit_idx) % 2) == 1

    def propagateShapes(self, make_symbolic=False):
        # Note: StridedSliceOp has many, many potential inputs and outputs,
        # so this function may only handle common cases
        # First input is the tensor to be sliced, second input is the begin
        # index, third input is the end index, and final input is the stride
        self.debugAssert(len(self._inputs) == 4)
        self.debugAssert(len(self._outputs) == 1)
        tensor_vals = self._inputs[0].value
        begin = self._inputs[1].value
        end = self._inputs[2].value
        stride = self._inputs[3].value
        if begin is None or end is None or stride is None:
            print('WARN: StridedSliceOp {} unable to resolve output shape'
                  .format(self._name))
            return
        if not isinstance(begin, list) and not isinstance(begin, np.ndarray):
            self.debugAssert(isinstance(begin, int))
            begin = [begin]
        if not isinstance(end, list) and not isinstance(begin, np.ndarray):
            self.debugAssert(isinstance(end, int))
            end = [end]
        if not isinstance(stride, list) and not isinstance(begin, np.ndarray):
            self.debugAssert(isinstance(stride, int))
            stride = [stride]

        if self._ellipsis_mask != 0:
            self.notImplemented('Handle ellipsis mask')
        if self._new_axis_mask != 0:
            self.notImplemented('Handle new axis mask')

        # Check input to output tensor shape propagation
        input_shape = self._inputs[0].shape
        out_dims = []
        for idx in range(input_shape.rank):
            if idx < len(begin):
                # Slice this dimension
                if (begin[idx] is not None or \
                    self.isIndexSet(self._begin_mask, idx)) and \
                   (end[idx] is not None or \
                    self.isIndexSet(self._end_mask, idx)) and \
                   stride[idx] is not None:
                    if self.isIndexSet(self._begin_mask, idx):
                        dim_begin = 0
                    else:
                        dim_begin = begin[idx]
                    if self.isIndexSet(self._end_mask, idx):
                        dim_end = input_shape.dims[idx].symbol
                    else:
                        dim_end = end[idx]
                    # Check if end dimension uses negative indexing, and if
                    # so, add the input tensor dimension to resolve positive
                    # index. NOTE: Must handle sympy symbols here:
                    # TODO: Move this dimension handling into Dimension class.
                    #       IOW, use all dimension handling functions here
                    check_dim_end_negative = dim_end < 0
                    try:
                        check_dim_end_negative = bool(check_dim_end_negative)
                    except:
                        pass
                    if isinstance(check_dim_end_negative, bool) and \
                       check_dim_end_negative:
                        dim_end += input_shape.dims[idx].symbol
                    dim_size = (dim_end - dim_begin + stride[idx] - 1)
                    dim_size //= stride[idx]
                    if self.isIndexSet(self._shrink_axis_mask, idx):
                        self.debugAssert(dim_size == 1)
                    else:
                        if dim_size == -1:
                            dim_size += input_shape.dims[idx].value
                        out_dims.append(dim_size)
                else:
                    self.notImplemented('Unspecified slice config')
            else:
                # If the ranges do not specify these dimensions, then
                # the input dimension gets preserved to the output
                out_dims.append(self._inputs[0].shape.getDimension(idx))
        self._outputs[0].mergeShape(out_dims,
                                    make_symbolic=make_symbolic)

        # Check if values can be resolved
        if not self._outputs[0].shape.isFullyNumeric() or tensor_vals is None:
            # Can only set up output values if the output shape is fully
            # resolved and the input tensor is available
            return

        if len(begin) == 1 and len(end) == 1 and len(stride) == 1:
            if self.isIndexSet(self._begin_mask, 0):
                dim_begin = None
            else:
                dim_begin = begin[0]
            if self.isIndexSet(self._end_mask, 0):
                dim_end = None
            else:
                dim_end = end[0]
            out_value = tensor_vals[dim_begin:dim_end:stride[0]]
        else:
            self.notImplemented('Unable to slice rank 2+ tensors')
        self._outputs[0].setValue(out_value)

    def calcAlgFlops(self):
        # StridedSliceOps have no Flops
        return 0

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class StridedSliceGradOp(Op):
    def __init__(self, name):
        super(StridedSliceGradOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        # Note: StridedSliceGradOp has many, many potential inputs and
        # outputs, so this function may only handle common cases
        # First input is the shape of the original tensor (unsliced). Inputs
        # second, third, and fourth are slicing config, fifth input is tensor
        # to unslice.
        self.debugAssert(len(self._inputs) == 5)
        self.debugAssert(len(self._outputs) == 1)
        if self._inputs[0].value is not None:
            out_dims = self._inputs[0].value
            self._outputs[0].mergeShape(out_dims,
                                        make_symbolic=make_symbolic)
        else:
            self.notImplemented('StridedSliceGradOp propagateShapes')

    def calcAlgFlops(self):
        # StridedSliceGradOps have no Flops
        return 0

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class TileOp(Op):
    def __init__(self, name):
        super(TileOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(len(self._inputs) == 2)
        self.debugAssert(len(self._outputs) == 1)

        # If input[0] has fully defined shape and input[1] has fully
        # defined value, then can set the output shape
        in_shape = self._inputs[0].shape
        if not in_shape.isFullySymbolic():
            # Cannot propagate shapes
            return
        multiples_val = self._inputs[1].value
        if multiples_val is None:
            # Cannot propagate shapes
            return

        out_shape = []
        for idx, dim in enumerate(in_shape.dims):
            out_shape.append(dim * multiples_val[idx])
        self._outputs[0].mergeShape(out_shape,
                                    make_symbolic=make_symbolic)

    def calcAlgFlops(self):
        # TileOps have no Flops
        return 0

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class TransposeOp(Op):
    def __init__(self, name):
        super(TransposeOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(len(self._inputs) == 2)
        self.debugAssert(len(self._outputs) == 1)
        if self._inputs[1].value is None:
            print('WARN: TransposeOp {} propagateShapes unknown input[1]'
                  .format(self._name))
            return

        # If second input has fully specified value, use it to propagate
        # first input dimensions to output dimensions
        permutation = self._inputs[1].value
        in_dims = list(self._inputs[0].shape.dims)
        out_dims = []
        for idx in permutation:
            out_dims.append(in_dims[idx])
        self._outputs[0].mergeShape(out_dims,
                                    make_symbolic=make_symbolic)

    def calcAlgFlops(self):
        # TransposeOps have no Flops
        return 0

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class WhereOp(Op):
    def __init__(self, name):
        super(WhereOp, self).__init__(name)
        # TODO: Make this parameterizable
        # When we assume that WhereOp subsamples the data, not all of the
        # input[0] values will exist in the output. The pessimistic view
        # is to assume that all input values will exist in the output
        # (i.e., self._use_subsampling = False)
        self._use_subsampling = False

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(len(self._outputs) == 1)
        if len(self._inputs) == 1:
            if self._inputs[0].shape.rank != 1:
                self.notImplemented('Where with rank {} input'
                                    .format(self._inputs[0].shape.rank))
            if self._use_subsampling:
                first_dim = utils.getIntSymbolFromString('{}::num_true'
                                .format(self._inputs[0].name))
            else:
                first_dim = self._inputs[0].shape.dims[0]
            self._outputs[0].mergeShape([first_dim, 1],
                                        make_symbolic=make_symbolic)
        else:
            self.debugAssert(len(self._inputs) == 3)
            self.notImplemented('Where with 2 inputs')

    def calcAlgFlops(self):
        self.debugAssert(len(self._inputs) == 1)
        # Assume one Flop per input element to check condition and decide
        # how to set the output
        flops = self._inputs[0].shape.numElements()
        return flops

    def calcAlgBytes(self):
        if len(self._inputs) == 1:
            input_bytes_accessed = self.bytesAccessInput()
        else:
            # Where ops access the second or third inputs conditioned on the
            # value supplied in the corresponding first input. Thus, the total
            # bytes accessed is only equal to two of the inputs plus the
            # output sizes
            self.debugAssert(len(self._inputs) == 3)
            input_bytes_accessed = self._inputs[0].size + self._inputs[1].size
            self.notImplemented('Untested: WhereOp calcAlgBytes 2+ inputs')
        return input_bytes_accessed + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class PackingOp(Op):
    def __init__(self, name):
        super(PackingOp, self).__init__(name)
        self._axis = None

    def setAxis(self, axis):
        self._axis = axis

    def calcAlgFlops(self):
        # Packing ops have no Flops
        return 0

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class PackOp(PackingOp):
    def __init__(self, name):
        super(PackOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(self._axis is not None)
        self.debugAssert(len(self._inputs) >= 1)
        in_shape = self._inputs[0].shape
        for in_tensor in self._inputs:
            self.debugAssert(in_tensor.shape == in_shape)

        # First, verify the output shape
        out_shape = list(in_shape.dims)
        num_ins = len(self._inputs)
        out_shape.insert(self._axis, Dimension(num_ins))
        self._outputs[0].mergeShape(out_shape,
                                    make_symbolic=make_symbolic)

        # If all input values exist, propagate them
        out_value = []
        can_prop_values = True
        for in_tensor in self._inputs:
            if in_tensor.value is not None:
                out_value.append(in_tensor.value)
            else:
                can_prop_values = False
                break

        if can_prop_values:
            self._outputs[0].setValue(out_value)


class UnpackOp(PackingOp):
    def __init__(self, name):
        super(UnpackOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(self._axis is not None)
        self.debugAssert(len(self._inputs) == 1)
        in_shape = self._inputs[0].shape
        num_outs = in_shape.dims[self._axis].value
        self.debugAssert(len(self._outputs) == num_outs)

        # First, verify output shapes
        out_shape = list(in_shape.dims)
        out_shape.pop(self._axis)
        for out_tensor in self._outputs:
            out_tensor.mergeShape(out_shape,
                                  make_symbolic=make_symbolic)

        # If values exist in input, propagate them
        if self._inputs[0].value is not None:
            in_vals = self._inputs[0].value
            if in_shape.rank == 1:
                for idx, out_tensor in enumerate(self._outputs):
                    out_tensor.setValue(in_vals[idx])
            else:
                self.notImplemented('UnpackOp propagateShapes: rank 2+')
