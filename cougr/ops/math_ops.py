import sympy

from .base_op import Op
from cougr.tensors.tensor import DataType


class BasePointwiseOp(Op):
    def __init__(self, name):
        super(BasePointwiseOp, self).__init__(name)
        self._flops_per_element = 1

    def propagateShapes(self):
        assert len(self._outputs) == 1
        if len(self._inputs) == 1:
            # Unary operator!
            final_shape = self._inputs[0].shape
        else:
            # Must be binary operator!
            assert len(self._inputs) == 2
            # Verify inputs have compatible shape
            in_a_shape = self._inputs[0].shape
            in_b_shape = self._inputs[1].shape
            assert not in_a_shape.isUnknown(), 'Op: {}'.format(self._name)
            assert not in_b_shape.isUnknown(), 'Op: {}'.format(self._name)
            assert in_a_shape.canBroadcastTogether(in_b_shape), \
                'Op: {}: input shapes cannot broadcast ({}, {})!'.format(
                self._name, in_a_shape, in_b_shape)
            final_shape = in_a_shape.getBroadcastShape(in_b_shape)

        # Set the output dimensions
        self._outputs[0].shape.mergeShape(final_shape)

    def flopsPointwise(self):
        assert len(self._outputs) == 1
        out_shape = self._outputs[0].shape
        return out_shape.numElements() * self._flops_per_element

    def calcAlgFlops(self):
        return self.flopsPointwise()


class AddOp(BasePointwiseOp):
    def __init__(self, name):
        super(AddOp, self).__init__(name)


class DivOp(BasePointwiseOp):
    def __init__(self, name):
        super(DivOp, self).__init__(name)


class ExpOp(BasePointwiseOp):
    def __init__(self, name):
        super(ExpOp, self).__init__(name)


class GreaterEqualOp(BasePointwiseOp):
    def __init__(self, name):
        super(GreaterEqualOp, self).__init__(name)


class LessOp(BasePointwiseOp):
    def __init__(self, name):
        super(LessOp, self).__init__(name)


class LogicalAndOp(BasePointwiseOp):
    def __init__(self, name):
        super(LogicalAndOp, self).__init__(name)


class LogicalNotOp(BasePointwiseOp):
    def __init__(self, name):
        super(LogicalNotOp, self).__init__(name)


class LogicalOrOp(BasePointwiseOp):
    def __init__(self, name):
        super(LogicalOrOp, self).__init__(name)


class MaximumOp(BasePointwiseOp):
    def __init__(self, name):
        super(MaximumOp, self).__init__(name)


class MinimumOp(BasePointwiseOp):
    def __init__(self, name):
        super(MinimumOp, self).__init__(name)


class MulOp(BasePointwiseOp):
    def __init__(self, name):
        super(MulOp, self).__init__(name)


class NegOp(BasePointwiseOp):
    def __init__(self, name):
        super(NegOp, self).__init__(name)


class NotEqualOp(BasePointwiseOp):
    def __init__(self, name):
        super(NotEqualOp, self).__init__(name)


class PowOp(BasePointwiseOp):
    def __init__(self, name):
        super(PowOp, self).__init__(name)


class ReluOp(BasePointwiseOp):
    def __init__(self, name):
        super(ReluOp, self).__init__(name)


class RsqrtOp(BasePointwiseOp):
    def __init__(self, name):
        super(RsqrtOp, self).__init__(name)
        # Assume reciprocal square root is 2 Flops per element
        self._flops_per_element = 2


class SigmoidOp(BasePointwiseOp):
    def __init__(self, name):
        super(SigmoidOp, self).__init__(name)
        # For now, assume sigmoid consists of input negation, exponentiation,
        # addition, and then inversion (i.e., 4 Flops per element)
        self._flops_per_element = 4


class SqrtOp(BasePointwiseOp):
    def __init__(self, name):
        super(SqrtOp, self).__init__(name)


class SubOp(BasePointwiseOp):
    def __init__(self, name):
        super(SubOp, self).__init__(name)


class TanhOp(BasePointwiseOp):
    def __init__(self, name):
        super(TanhOp, self).__init__(name)
        # For now, assume tanh consists of input negation, two
        # exponentiations, addition and subtraction, and then inversion
        # (i.e., 6 Flops per element)
        self._flops_per_element = 6


class Conv2DOp(Op):
    def __init__(self, name):
        super(Conv2DOp, self).__init__(name)

    def calcAlgFlops(self):
        raise NotImplementedError('Conv2D alg Flops not implemented yet!')


class MatMulOp(Op):
    def __init__(self, name):
        super(MatMulOp, self).__init__(name)
        self._transpose_a = False
        self._transpose_b = False
        self._transpose_c = False

    def setTransposeInput(self, input_idx, is_transposed):
        if input_idx == 0:
            self._transpose_a = is_transposed
        elif input_idx == 1:
            self._transpose_b = is_transposed
        else:
            assert input_idx == 0 or input_idx == 1, \
                'Unknown input index {}'.format(input_idx)

    def propagateShapes(self):
        # [_] TODO (Joel): Will need to handle transposes here...
        # Verify that shapes can be correctly resolved
        assert len(self._inputs) == 2
        tensor_a = self._inputs[0]
        tensor_b = self._inputs[1]
        if not self._transpose_a:
            first_dim = tensor_a.shape.getDimension(0)
            inner_dim = tensor_a.shape.getDimension(1)
        else:
            first_dim = tensor_a.shape.getDimension(1)
            inner_dim = tensor_a.shape.getDimension(0)
        if not self._transpose_b:
            b_in_dim = tensor_b.shape.getDimension(0)
            last_dim = tensor_b.shape.getDimension(1)
        else:
            b_in_dim = tensor_b.shape.getDimension(1)
            last_dim = tensor_b.shape.getDimension(0)
        # [_] TODO (Joel): This assert will be too strict at some point
        import sympy
        assert (isinstance(inner_dim, sympy.Symbol) or \
                isinstance(b_in_dim, sympy.Symbol) or \
                inner_dim == b_in_dim), \
               'Dimension check failed in op {} with inputs {} and {}, '\
               'and output {}'.format(self._name, self._inputs[0].shape,
                                      self._inputs[1].shape,
                                      self._outputs[0].shape)
        # Resolve shapes
        if self._transpose_c:
            raise NotImplementedError('Handle transposed output')
        tensor_c = self._outputs[0]
        tensor_c.shape.mergeShape([first_dim, last_dim])

    def calcAlgFlops(self):
        # [_] TODO (Joel): Will need to handle transposes here...
        # Get matrix inner dimension
        assert len(self._inputs) == 2
        tensor_a = self._inputs[0]
        tensor_b = self._inputs[1]
        if not self._transpose_a:
            inner_dim = tensor_a.shape.getDimension(1)
        else:
            inner_dim = tensor_a.shape.getDimension(0)
        if not self._transpose_b:
            b_in_dim = tensor_b.shape.getDimension(0)
        else:
            b_in_dim = tensor_b.shape.getDimension(1)
        # [_] TODO (Joel): This assert will be too strict at some point
        import sympy
        assert (isinstance(inner_dim, sympy.Symbol) or \
                isinstance(b_in_dim, sympy.Symbol) or \
                inner_dim == b_in_dim), \
               'Dimension check failed in op {} with inputs {} and {}, '\
               'and output {}'.format(self._name, self._inputs[0].shape,
                                      self._inputs[1].shape,
                                      self._outputs[0].shape)
        # Get output number of elements
        if self._transpose_c:
            raise NotImplementedError('Handle transposed output')
        assert len(self._outputs) == 1
        out_shape = self._outputs[0].shape
        out_elts = out_shape.numElements()
        return (2 * inner_dim.symbol * out_elts)


class RangeOp(Op):
    def __init__(self, name):
        super(RangeOp, self).__init__(name)

    def propagateShapes(self):
        assert len(self._inputs) == 3
        assert len(self._outputs) == 1
        start = self._inputs[0].value
        limit = self._inputs[1].value
        delta = self._inputs[2].value

        if start is None or limit is None or delta is None:
            return

        # Set output shape
        supported_shape_types = (int, sympy.Symbol)
        if self._outputs[0].dtype == DataType.int32:
            if isinstance(start, supported_shape_types) and \
               isinstance(limit, supported_shape_types) and \
               isinstance(delta, supported_shape_types):
                num_elts = (limit - start + delta - 1) // delta
                self._outputs[0].shape.mergeShape([num_elts])
            else:
                self.notImplemented('RangeOp shape types {} {} {}'
                                    .format(start, limit, delta))
        else:
            self.notImplemented('RangeOp other data type {}'
                                .format(self._outputs[0].dtype))

        if not isinstance(start, int) or not isinstance(limit, int) or \
           not isinstance(delta, int):
            return

        value = [val for val in range(start, limit, delta)]
        # Note: Setting output value will verify that the shapes are
        # fully specified and match
        self._outputs[0].setValue(value)

    def calcAlgFlops(self):
        # Range op has no algorithmic Flops
        return 0


class ReduceOp(Op):
    def __init__(self, name, reduction_op='sum', axes=None):
        super(ReduceOp, self).__init__(name)
        if isinstance(axes, int):
            axes = [axes]
        self._axes = axes
        self._flops_per_element = 1

    def propagateShapes(self):
        # First input is always the tensor to be reduce, and the optional
        # second tensor describes the dimensions to be reduced.
        assert len(self._inputs) == 1 or len(self._inputs) == 2
        assert len(self._outputs) == 1
        if len(self._inputs) == 2:
            # TODO (Joel): May be too strict
            assert self._axes is None
            self._axes = self._inputs[1].value
        if self._axes is not None:
            out_shape = []
            for dim_index in range(self._inputs[0].shape.rank):
                if dim_index not in self._axes:
                    dim = self._inputs[0].shape.getDimension(dim_index)
                    out_shape.append(dim)
            self._outputs[0].shape.mergeShape(out_shape)
            # TODO (Joel): Also calculate value? Depends on the reduction_op!

    def calcAlgFlops(self):
        assert len(self._inputs) == 1 or len(self._inputs) == 2, \
            'Reduce {} has too many inputs: {}' \
            .format(self.name, [input for input in self._inputs])
        flops_to_return = self._flops_per_element
        for dim_index in range(self._inputs[0].shape.rank):
            dim = self._inputs[0].shape.getDimension(dim_index)
            flops_to_return *= dim.symbol
        return flops_to_return

