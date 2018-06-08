from .base_op import Op
from ..tensors.tensor_shape import Dimension, TensorShape

class ConcatOp(Op):
    def __init__(self, name):
        super(ConcatOp, self).__init__(name)

    def propagateShapes(self):
        # Verify input shapes can be merged when concat axis is masked
        assert len(self._inputs) >= 2
        assert self._inputs[-1].shape.rank == 0
        axis = self._inputs[-1].value
        assert axis is not None, 'Op {} axis is None'.format(self._name)
        assert len(self._outputs) == 1
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
            self._outputs[0].shape.mergeShape(in_shape)
        self._outputs[0].shape.setDimension(axis, out_shape_c_dim)

        if not propagate_values:
            return

        # If the input values are fully specified, then also propagate
        # those values to the outputs
        out_value = None
        for idx in range(len(self._inputs) - 1):
            if self._inputs[idx].shape.rank != 1 or axis != 0:
                raise NotImplementedError('ConcatOp {} values non-1 rank'
                                          .format(self._name))
            if out_value is None:
                out_value = self._inputs[idx].value
            else:
                out_value.extend(self._inputs[idx].value)
        self._outputs[0].setValue(out_value)

    def calcAlgFlops(self):
        # ConcatOps have no Flops
        return 0


class ExpandDimsOp(Op):
    def __init__(self, name):
        super(ExpandDimsOp, self).__init__(name)

    def propagateShapes(self):
        assert len(self._inputs) == 2
        assert len(self._outputs) == 1
        assert self._outputs[0].shape.isFullyDefined()
        axis = self._inputs[1].value
        if axis is None:
            print('WARN: Undetermined axis for ExpandDimsOp {}'
                  .format(self._name))
            return
        if self._inputs[0].shape.rank == 0:
            assert axis == 0
            out_value = [self._inputs[0].value]
            self._outputs[0].setValue(out_value)
        else:
            raise NotImplementedError(
                'ExpandDimsOp propagateShapes rank 1+')

    def calcAlgFlops(self):
        # ExpandDimsOp has no algorithmic Flops
        return 0


class FillOp(Op):
    ''' Returns a tensor of the shape specified in first input tensor (and
        fills that tenros with a set value passed in second input).
        in each location of the output tensor. Similar to Numpy and TF
        Fill.
    '''
    def __init__(self, name):
        super(FillOp, self).__init__(name)

    def propagateShapes(self):
        # FillOps should create an output tensor with the shape passed
        # as the first input tensor
        assert len(self._inputs) == 2, 'FillOp {} has {} inputs' \
            .format(self._name, len(self._inputs))
        assert len(self._outputs) == 1, 'FillOp {} has {} outputs' \
            .format(self._name, len(self._outputs))
        # The first input tensor contains the shape of output tensor
        out_shape = self._inputs[0].value
        if out_shape is None:
            print('WARN: FillOp {} shape undetermined'.format(self._name))
            return
        self._outputs[0].shape.mergeShape(out_shape)
        if self._outputs[0].shape.isFullyDefined():
            raise NotImplementedError('FillOp may specify output value')

    def calcAlgFlops(self):
        # FillOps have no Flops
        return 0


class GatherOp(Op):
    def __init__(self, name):
        super(GatherOp, self).__init__(name)

    def propagateShapes(self):
        # First input is the tensor from which to gather. Second input is
        # the indices to gather from first tensor.
        assert len(self._inputs) == 2
        assert len(self._outputs) == 1
        # TODO (Joel): Optional third tensor can specify the axis of data
        # to index.

    def calcAlgFlops(self):
        # GatherOps have no Flops
        return 0


class NumLikeOp(Op):
    ''' Returns a tensor of the same shape as input with a set value
        in each location of the output tensor. Similar to Numpy and TF
        ZerosLike and OnesLike.
    '''
    def __init__(self, name):
        super(NumLikeOp, self).__init__(name)

    def propagateShapes(self):
        # NumLikeOps should create an output tensor with the shape passed
        # as the first input tensor
        assert len(self._inputs) == 1, 'NumLikeOp {} has {} inputs' \
            .format(self._name, len(self._inputs))
        assert len(self._outputs) == 1, 'NumLikeOp {} has {} outputs' \
            .format(self._name, len(self._outputs))
        # The output tensor should be the same shape as the input
        if not self._inputs[0].shape.isUnknown():
            if self._inputs[0].shape != self._outputs[0].shape:
                raise NotImplementedError('NumLikeOp propagateShapes {}'
                                          .format(self._name))
            self._outputs[0].shape.mergeShape(self._inputs[0].shape)

    def calcAlgFlops(self):
        # NumLikeOps have no Flops
        return 0


class ReshapeOp(Op):
    def __init__(self, name):
        super(ReshapeOp, self).__init__(name)

    def propagateShapes(self):
        # The first input is reshaped according to the second tensor
        # 1) If the second tensor is [], try to reshape the input to a scalar
        # 2) If the second tensor contains -1, then reshape according to the
        #    other dimensions of second tensor and fill the -1 dimension to
        #    maintain the same number of elements.
        print('WARN: ReshapeOp propagateShapes not complete')
        pass

    def calcAlgFlops(self):
        # ReshapeOps have no Flops
        return 0


class ScatterOp(Op):
    def __init__(self, name):
        super(ScatterOp, self).__init__(name)

    def calcAlgFlops(self):
        # ScatterOps have no Flops
        return 0


class ShapeOp(Op):
    def __init__(self, name):
        super(ShapeOp, self).__init__(name)

    def propagateShapes(self):
        # The output shape should be a 1D tensor with specified dimension
        # equal to the rank of the input tensor
        assert len(self._inputs) == 1
        out_rank = self._inputs[0].shape.rank
        assert len(self._outputs) == 1
        self._outputs[0].shape.setDimension(0, out_rank)

        # Can set some output values if the input shape is known
        if not self._inputs[0].shape.isUnknown():
            out_value = []
            for dim in self._inputs[0].shape.dims:
                # Can propagate symbols and values, so request symbols
                out_value.append(dim.symbol)
            self._outputs[0].setValue(out_value)

    def calcAlgFlops(self):
        # ShapeOps have no Flops
        return 0


class SizeOp(Op):
    def __init__(self, name):
        super(SizeOp, self).__init__(name)

    def propagateShapes(self):
        assert len(self._inputs) == 1
        assert len(self._outputs) == 1
        if self._outputs[0].shape.isUnknown():
            self._outputs[0].shape.mergeShape([])
        else:
            assert self._outputs[0].shape.isScalar()
        self._outputs[0].setValue(self._inputs[0].shape.numElements())

    def calcAlgFlops(self):
        # SizeOps have no Flops
        return 0


class SplitOp(Op):
    def __init__(self, name, num_splits=0, axis=0):
        super(SplitOp, self).__init__(name)

    def propagateShapes(self):
        # Assume SliceOps have fully specified shapes
        pass

    def calcAlgFlops(self):
        # SplitOps have no Flops
        return 0


class StackOp(Op):
    def __init__(self, name):
        super(StackOp, self).__init__(name)

    def propagateShapes(self):
        # Assume StackOps have fully specified shapes
        pass

    def calcAlgFlops(self):
        # StackOps have no Flops
        return 0


class StridedSliceOp(Op):
    def __init__(self, name):
        super(StridedSliceOp, self).__init__(name)

    def propagateShapes(self):
        # Note: StridedSliceOp has many, many potential inputs and outputs,
        # so this function may only handle common cases
        # First input is the tensor to be sliced, second input is the begin
        # index, third input is the end index, and final input is the stride
        assert len(self._inputs) == 4
        assert len(self._outputs) == 1
        tensor_vals = self._inputs[0].value
        begin = self._inputs[1].value
        end = self._inputs[2].value
        stride = self._inputs[3].value
        if begin is None or end is None or stride is None:
            print('WARN: StridedSliceOp {} unable to resolve output shape'
                  .format(self._name))
            return
        if not isinstance(begin, list):
            assert isinstance(begin, int)
            begin = [begin]
        if not isinstance(end, list):
            assert isinstance(end, int)
            end = [end]
        if not isinstance(stride, list):
            assert isinstance(stride, int)
            stride = [stride]

        # Check input to output tensor shape propagation
        if not self._outputs[0].shape.isFullyDefined():
            out_dims = []
            for idx in range(self._inputs[0].shape.rank):
                if idx < len(begin):
                    # Slice this dimension
                    if begin[idx] is not None and \
                       end[idx] is not None and \
                       stride[idx] is not None:
                        dim_size = (end[idx] - begin[idx] + stride[idx] - 1)
                        dim_size //= stride[idx]
                        if dim_size > 1:
                            out_dims.append(dim_size)
                    else:
                        raise NotImplementedError('Unspecified slice config')
                else:
                    # If the ranges to not specify these dimensions, then
                    # the input dimension gets preserved to the output
                    out_dims.append(self._inputs[0].shape.dims[idx])
            self._outputs[0].shape.mergeShape(out_dims)

        # Check if values can be resolved
        if not self._outputs[0].shape.isFullyDefined() or tensor_vals is None:
            # Can only set up output values if the output shape is fully
            # resolved and the input tensor is available
            return

        if len(begin) == 1 and len(end) == 1 and len(stride) == 1:
            out_value = tensor_vals[begin[0]:end[0]:stride[0]]
        else:
            raise NotImplementedError('Unable to slice rank 2+ tensors')
        self._outputs[0].setValue(out_value)

    def calcAlgFlops(self):
        # StridedSliceOps have no Flops
        return 0


class TensorArrayOp(Op):
    def __init__(self, name):
        super(TensorArrayOp, self).__init__(name)

    def propagateShapes(self):
        # TODO (Joel): Propagate shapes for TensorArrayOp
        pass

    def calcAlgFlops(self):
        # TensorArrayOps only allow creation of tensor arrays
        return 0


class TransposeOp(Op):
    def __init__(self, name):
        super(TransposeOp, self).__init__(name)

    def propagateShapes(self):
        assert len(self._inputs) == 2
        assert len(self._outputs) == 1
        if self._inputs[1].value is None:
            raise NotImplementedError(
                'TransposeOp propagateShapes: try to infer shapes')

        # If second input has fully specified value, use it to propagate
        # first input dimensions to output dimensions
        permutation = self._inputs[1].value
        in_dims = list(self._inputs[0].shape.dims)
        out_dims = []
        for idx in permutation:
            out_dims.append(in_dims[idx])
        self._outputs[0].shape.mergeShape(out_dims)

    def calcAlgFlops(self):
        # TransposeOps have no Flops
        return 0


