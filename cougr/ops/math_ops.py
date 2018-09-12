import numpy as np
import sympy

from .base_op import Op
from cougr.tensors.tensor import DataType
from cougr.tensors.tensor_shape import TensorShape


def as_int(value):
    if isinstance(value, int):
        return value
    elif isinstance(value, sympy.Expr):
        assert value.is_integer
        return value
    elif isinstance(value, np.int64):
        return int(value)
    else:
        raise TypeError('Unknown int type ({}) for value {}'
                        .format(type(value), value))

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
            self.debugAssert(not in_a_shape.isUnknown(),
                             'Op: {}'.format(self._name))
            self.debugAssert(not in_b_shape.isUnknown(),
                             'Op: {}'.format(self._name))
            self.debugAssert(in_a_shape.canBroadcastTogether(in_b_shape),
                'Op: {}: input shapes cannot broadcast ({}, {})!'.format(
                self._name, in_a_shape, in_b_shape))
            final_shape = in_a_shape.getBroadcastShape(in_b_shape)

        # Set the output dimensions
        if not final_shape.isUnknown():
            self._outputs[0].shape.mergeShape(final_shape)

    def flopsPointwise(self):
        self.debugAssert(len(self._outputs) == 1)
        out_shape = self._outputs[0].shape
        return out_shape.numElements() * self._flops_per_element

    def calcAlgFlops(self):
        return self.flopsPointwise()

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class AddOp(BasePointwiseOp):
    def __init__(self, name):
        super(AddOp, self).__init__(name)

    def propagateShapes(self):
        super(AddOp, self).propagateShapes()
        self.debugAssert(len(self._inputs) == 2)
        if self._inputs[0].value is not None and \
           self._inputs[1].value is not None:
            self._outputs[0].setValue(np.add(self._inputs[0].value,
                                             self._inputs[1].value))


class AddNOp(Op):
    def __init__(self, name):
        super(AddNOp, self).__init__(name)

    def propagateShapes(self):
        self.debugAssert(len(self._outputs) == 1)
        in_shape = None
        for in_tensor in self._inputs:
            if in_shape is None:
                if not in_tensor.shape.isUnknown():
                    in_shape = in_tensor.shape
            else:
                if not in_tensor.shape.isUnknown():
                    self.debugAssert(in_shape == in_tensor.shape)
        if in_shape is not None:
            self._outputs[0].shape.mergeShape(in_shape)

    def calcAlgFlops(self):
        self.debugAssert(len(self._outputs) == 1)
        num_inputs = len(self._inputs)
        out_shape = self._outputs[0].shape
        return out_shape.numElements() * (num_inputs - 1)

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class DivOp(BasePointwiseOp):
    def __init__(self, name):
        super(DivOp, self).__init__(name)


class EqualOp(BasePointwiseOp):
    def __init__(self, name):
        super(EqualOp, self).__init__(name)


class ExpOp(BasePointwiseOp):
    def __init__(self, name):
        super(ExpOp, self).__init__(name)


class FloorDivOp(BasePointwiseOp):
    def __init__(self, name):
        super(FloorDivOp, self).__init__(name)

    def propagateShapes(self):
        super(FloorDivOp, self).propagateShapes()
        self.debugAssert(len(self._inputs) == 2)
        if self._inputs[0].value is not None and \
           self._inputs[1].value is not None:
            self._outputs[0].setValue(np.floor_divide(
                self._inputs[0].value, self._inputs[1].value))


class FloorModOp(BasePointwiseOp):
    def __init__(self, name):
        super(FloorModOp, self).__init__(name)

    def propagateShapes(self):
        super(FloorModOp, self).propagateShapes()
        self.debugAssert(len(self._inputs) == 2)
        if self._inputs[0].value is not None and \
           self._inputs[1].value is not None:
            self._outputs[0].setValue(np.mod(self._inputs[0].value,
                                             self._inputs[1].value))


class GreaterOp(BasePointwiseOp):
    def __init__(self, name):
        super(GreaterOp, self).__init__(name)


class GreaterEqualOp(BasePointwiseOp):
    def __init__(self, name):
        super(GreaterEqualOp, self).__init__(name)

    def propagateShapes(self):
        super(GreaterEqualOp, self).propagateShapes()
        self.debugAssert(len(self._inputs) == 2)
        if self._inputs[0].value is None or \
           self._inputs[1].value is None:
            return
        in_0_val = self._inputs[0].value
        in_1_val = self._inputs[1].value
        out_val = in_0_val >= in_1_val
        self._outputs[0].setValue(out_val)


class LessOp(BasePointwiseOp):
    def __init__(self, name):
        super(LessOp, self).__init__(name)

    def propagateShapes(self):
        super(LessOp, self).propagateShapes()
        self.debugAssert(len(self._inputs) == 2)
        if self._inputs[0].value is None or \
           self._inputs[1].value is None:
            return
        in_0_val = self._inputs[0].value
        in_1_val = self._inputs[1].value
        out_val = in_0_val < in_1_val
        self._outputs[0].setValue(out_val)


class LogOp(BasePointwiseOp):
    def __init__(self, name):
        super(LogOp, self).__init__(name)


class LogicalAndOp(BasePointwiseOp):
    def __init__(self, name):
        super(LogicalAndOp, self).__init__(name)


class LogicalOrOp(BasePointwiseOp):
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

    def propagateShapes(self):
        super(MaximumOp, self).propagateShapes()
        self.debugAssert(len(self._inputs) == 2)
        # TODO (Joel): Move the functor/lambda definition out to be
        # a class member, and let BasePointwiseOp apply if not None
        # if self._inputs[0].value is not None and \
        #    self._inputs[1].value is not None:
        #     vmax = np.vectorize(lambda x, y: sympy.Max(x, y))
        #     out_val = vmax(self._inputs[0].value, self._inputs[1].value)
        #     self._outputs[0].setValue(out_val)


class MinimumOp(BasePointwiseOp):
    def __init__(self, name):
        super(MinimumOp, self).__init__(name)


class MulOp(BasePointwiseOp):
    def __init__(self, name):
        super(MulOp, self).__init__(name)

    def propagateShapes(self):
        super(MulOp, self).propagateShapes()
        self.debugAssert(len(self._inputs) == 2)
        if self._inputs[0].value is not None and \
           self._inputs[1].value is not None:
            self._outputs[0].setValue(np.multiply(self._inputs[0].value,
                                                  self._inputs[1].value))


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
        # Relu just makes a single comparison


class ReluGradOp(BasePointwiseOp):
    def __init__(self, name):
        super(ReluGradOp, self).__init__(name)
        # ReluGrad just makes a single comparison


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


class SigmoidGradOp(Op):
    ''' Computes the gradient of the sigmoid of `x` wrt its input.
    Specifically, `grad = dy * y * (1 - y)`, where `y = sigmoid(x)`, and
    `dy` is the corresponding input gradient.
    '''
    def __init__(self, name):
        super(SigmoidGradOp, self).__init__(name)

    def propagateShapes(self):
        self.debugAssert(len(self._inputs) == 2)
        self.debugAssert(len(self._outputs) == 1)
        in_shape = None
        if not self._inputs[0].shape.isUnknown():
            in_shape = self._inputs[0].shape
        if not self._inputs[1].shape.isUnknown():
            if in_shape is None:
                in_shape = self._inputs[1].shape
            else:
                self.debugAssert(in_shape == self._inputs[1].shape)
        self._outputs[0].shape.mergeShape(in_shape)

    def calcAlgFlops(self):
        self.debugAssert(len(self._inputs) == 2)
        self.debugAssert(len(self._outputs) == 1)
        in_shape = None
        if not self._inputs[0].shape.isUnknown():
            in_shape = self._inputs[0].shape
        if not self._inputs[1].shape.isUnknown():
            if in_shape is None:
                in_shape = self._inputs[1].shape
            else:
                self.debugAssert(in_shape == self._inputs[1].shape)
        if in_shape is None:
            if not self._outputs[0].shape.isUnknown():
                in_shape = self._outputs[0].shape.isUnknown()
            else:
                self.debugAssert(in_shape == self._outputs[0].shape)
        return 3 * in_shape.numElements()

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class SqrtOp(BasePointwiseOp):
    def __init__(self, name):
        super(SqrtOp, self).__init__(name)


class SubOp(BasePointwiseOp):
    def __init__(self, name):
        super(SubOp, self).__init__(name)

    def propagateShapes(self):
        super(SubOp, self).propagateShapes()
        self.debugAssert(len(self._inputs) == 2)
        if self._inputs[0].value is not None and \
           self._inputs[1].value is not None:
            # TODO (Joel): Consider wrapping this in try-catch in case some
            # portion of value is None that would cause crashes
            out_value = self._inputs[0].value - self._inputs[1].value
            self._outputs[0].setValue(out_value)


class TanhOp(BasePointwiseOp):
    def __init__(self, name):
        super(TanhOp, self).__init__(name)
        # For now, assume tanh consists of input negation, two
        # exponentiations, addition and subtraction, and then inversion
        # (i.e., 6 Flops per element)
        self._flops_per_element = 6


class TanhGradOp(Op):
    ''' Computes the gradient for the tanh of `x` wrt its input.
    Specifically, `grad = dy * (1 - y*y)`, where `y = tanh(x)`, and `dy`
    is the corresponding input gradient.
    '''
    def __init__(self, name):
        super(TanhGradOp, self).__init__(name)

    def propagateShapes(self):
        self.debugAssert(len(self._inputs) == 2)
        self.debugAssert(len(self._outputs) == 1)
        in_shape = None
        if not self._inputs[0].shape.isUnknown():
            in_shape = self._inputs[0].shape
        if not self._inputs[1].shape.isUnknown():
            if in_shape is None:
                in_shape = self._inputs[1].shape
            else:
                self.debugAssert(in_shape == self._inputs[1].shape)
        self._outputs[0].shape.mergeShape(in_shape)

    def calcAlgFlops(self):
        self.debugAssert(len(self._inputs) == 2)
        self.debugAssert(len(self._outputs) == 1)
        in_shape = None
        if not self._inputs[0].shape.isUnknown():
            in_shape = self._inputs[0].shape
        if not self._inputs[1].shape.isUnknown():
            if in_shape is None:
                in_shape = self._inputs[1].shape
            else:
                self.debugAssert(in_shape == self._inputs[1].shape)
        if in_shape is None:
            if not self._outputs[0].shape.isUnknown():
                in_shape = self._outputs[0].shape.isUnknown()
            else:
                self.debugAssert(in_shape == self._outputs[0].shape)
        return 3 * in_shape.numElements()

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class Conv2DBaseOp(Op):
    def __init__(self, name):
        super(Conv2DBaseOp, self).__init__(name)
        self._format = None
        self._strides = None
        self._dilations = None

    def setDataFormat(self, format):
        if format != 'NCHW' and format != 'NHWC':
            self.notImplemented('Unknown data format: {}'.format(format))
        self._format = format

    def setStrides(self, strides):
        self.debugAssert(isinstance(strides, list))
        self._strides = strides

    def setDilations(self, dilations):
        self.debugAssert(isinstance(dilations, list))
        for val in dilations:
            if val != 1:
                self.notImplemented('Unhandled Conv2D dilation: {}'
                                    .format(dilations))
        self._dilations = dilations


    def calcConv2DFlops(self, filter_shape, output_shape, backprop=False):
        filter_height = filter_shape.getDimension(0).symbol
        filter_width = filter_shape.getDimension(1).symbol
        if backprop:
            filter_depth = filter_shape.getDimension(3).symbol
        else:
            filter_depth = filter_shape.getDimension(2).symbol
        output_size = output_shape.numElements()
        flops = output_size * filter_depth * filter_height * \
                filter_width * 2
        return flops

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class Conv2DGradFilterOp(Conv2DBaseOp):
    def __init__(self, name):
        super(Conv2DGradFilterOp, self).__init__(name)

    def propagateShapes(self):
        self.debugAssert(len(self._inputs) == 3)
        self.debugAssert(len(self._outputs) == 1)
        self.debugAssert(self._strides is not None)

        self.debugAssert(self._inputs[1].value is not None)
        filter_shape = self._inputs[1].value.tolist()
        self._outputs[0].shape.mergeShape(filter_shape)

    def calcAlgFlops(self):
        self.debugAssert(len(self._inputs) == 3)
        self.debugAssert(len(self._outputs) == 1)
        self.debugAssert(self._outputs[0].shape.rank == 4)

        # This op's inputs are the output shape of the forward-prop op
        output_shape = self._inputs[0].shape
        grads_shape = self._inputs[2].shape
        if self._format == 'NCHW':
            in_chans = output_shape.getDimension(1).symbol
            out_chans = grads_shape.getDimension(1).symbol
            stride_height = self._strides[2]
            stride_width = self._strides[3]
        elif self._format == 'NHWC':
            in_chans = output_shape.getDimension(3).symbol
            out_chans = grads_shape.getDimension(3).symbol
            stride_height = self._strides[1]
            stride_width = self._strides[2]
        else:
            self.notImplemented('Unknown data format: {}'
                                .format(self._format))
        # To backprop to the filter, the op's output is the filter's shape
        filter_shape = self._outputs[0].shape
        f_in_chans = filter_shape.getDimension(2).symbol
        f_out_chans = filter_shape.getDimension(3).symbol
        self.debugAssert(f_in_chans - in_chans == 0)
        self.debugAssert(f_out_chans - out_chans == 0,
                'Conv2DGradFilter Dims differ: f_out_chans: {}, out_chans: {}'
                .format(f_out_chans, out_chans))
        flops = self.calcConv2DFlops(filter_shape, output_shape,
                                     backprop=True)
        flops /= (stride_height * stride_width)
        return flops


class Conv2DGradInputOp(Conv2DBaseOp):
    def __init__(self, name):
        super(Conv2DGradInputOp, self).__init__(name)

    def propagateShapes(self):
        self.debugAssert(len(self._inputs) == 3)
        self.debugAssert(len(self._outputs) == 1)
        self.debugAssert(self._strides is not None)

        grads_shape = self._inputs[2].shape
        batch_size = grads_shape.getDimension(0)
        filter_shape = self._inputs[1].shape
        in_chans = filter_shape.getDimension(2)
        if self._format == 'NCHW':
            if self._inputs[0].value is not None:
                # If tracked, prefer to directly pull shape from input
                in_height = self._inputs[0].value[2]
                in_width = self._inputs[0].value[3]
            else:
                out_height = grads_shape.getDimension(2)
                out_width = grads_shape.getDimension(3)
                in_height = out_height * self._strides[2]
                in_width = out_width * self._strides[3]
            out_shape = [batch_size, in_chans, in_height, in_width]
        elif self._format == 'NHWC':
            if self._inputs[0].value is not None:
                # If tracked, prefer to directly pull shape from input
                in_height = self._inputs[0].value[1]
                in_width = self._inputs[0].value[2]
            else:
                out_height = grads_shape.getDimension(1)
                out_width = grads_shape.getDimension(2)
                in_height = out_height * self._strides[1]
                in_width = out_width * self._strides[2]
            out_shape = [batch_size, in_height, in_width, in_chans]
        else:
            self.notImplemented('Unknown data format: {}'
                                .format(self._format))
        self._outputs[0].shape.mergeShape(out_shape)

    def calcAlgFlops(self):
        self.debugAssert(len(self._inputs) == 3)
        self.debugAssert(len(self._outputs) == 1)
        self.debugAssert(self._outputs[0].shape.rank == 4)

        output_shape = self._outputs[0].shape
        grads_shape = self._inputs[2].shape
        if self._format == 'NCHW':
            in_chans = output_shape.getDimension(1).symbol
            out_chans = grads_shape.getDimension(1).symbol
            stride_height = self._strides[2]
            stride_width = self._strides[3]
        elif self._format == 'NHWC':
            in_chans = output_shape.getDimension(3).symbol
            out_chans = grads_shape.getDimension(3).symbol
            stride_height = self._strides[1]
            stride_width = self._strides[2]
        else:
            self.notImplemented('Unknown data format: {}'
                                .format(self._format))
        filter_shape = self._inputs[1].shape
        f_in_chans = filter_shape.getDimension(2).symbol
        f_out_chans = filter_shape.getDimension(3).symbol
        if isinstance(f_in_chans, int) and isinstance(in_chans, int):
            self.debugAssert(f_in_chans - in_chans == 0,
                'Conv2D Dims differ: f_in_chans: {}, in_chans: {}'
                .format(f_in_chans, in_chans))
        elif isinstance(f_in_chans, (sympy.Symbol, sympy.Expr)) and \
            isinstance(in_chans, (sympy.Symbol, sympy.Expr)):
            self.debugAssert((f_in_chans - in_chans).simplify() == 0,
                'Conv2D Dims differ: f_in_chans: {}, in_chans: {}'
                .format(f_in_chans, in_chans))
        else:
            print('WARN: Conv2D dimension types: f_in_chans: {}, in_chans: {}'
                  .format(f_in_chans, in_chans))
        self.debugAssert(f_out_chans - out_chans == 0)
        flops = self.calcConv2DFlops(filter_shape, output_shape,
                                     backprop=True)
        flops /= (stride_height * stride_width)
        return flops


class Conv2DOp(Conv2DBaseOp):
    def __init__(self, name):
        super(Conv2DOp, self).__init__(name)

    def propagateShapes(self):
        self.debugAssert(len(self._inputs) == 2)
        self.debugAssert(len(self._outputs) == 1)
        self.debugAssert(self._strides is not None)

        in_shape = self._inputs[0].shape
        batch_size = in_shape.getDimension(0)
        filter_shape = self._inputs[1].shape
        out_chans = filter_shape.getDimension(3)
        if self._format == 'NCHW':
            in_height = in_shape.getDimension(2)
            in_width = in_shape.getDimension(3)
            out_height = in_height // self._strides[2]
            out_width = in_width // self._strides[3]
            out_shape = [batch_size, out_chans, out_height, out_width]
        elif self._format == 'NHWC':
            in_height = in_shape.getDimension(1)
            in_width = in_shape.getDimension(2)
            out_height = in_height // self._strides[1]
            out_width = in_width // self._strides[2]
            out_shape = [batch_size, out_height, out_width, out_chans]
        else:
            self.notImplemented('Unknown data format: {}'
                                .format(self._format))
        self._outputs[0].shape.mergeShape(out_shape)

    def calcAlgFlops(self):
        self.debugAssert(len(self._inputs) == 2)
        self.debugAssert(len(self._outputs) == 1)

        output_shape = self._outputs[0].shape
        if self._format == 'NCHW':
            out_chans = output_shape.getDimension(1).symbol
        elif self._format == 'NHWC':
            out_chans = output_shape.getDimension(3).symbol
        else:
            self.notImplemented('Unknown data format: {}'
                                .format(self._format))
        filter_shape = self._inputs[1].shape
        filter_count = filter_shape.getDimension(3).symbol
        self.debugAssert(filter_count - out_chans == 0)
        flops = self.calcConv2DFlops(filter_shape, output_shape)
        return flops


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

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class PoolBaseOp(Op):
    def __init__(self, name):
        super(PoolBaseOp, self).__init__(name)
        self._format = None
        self._ksize = None
        self._strides = None

    def setDataFormat(self, format):
        if format != 'NCHW' and format != 'NHWC':
            self.notImplemented('Unknown data format: {}'.format(format))
        self._format = format

    def setKSize(self, ksize):
        self.debugAssert(isinstance(ksize, list), 'setKSize to {} of type {}'
                         .format(ksize, type(ksize)))
        self._ksize = ksize

    def setStrides(self, strides):
        self.debugAssert(isinstance(strides, list),
                         'setStrides to {} of type {}'
                         .format(strides, type(strides)))
        self._strides = strides

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class MaxPoolOp(PoolBaseOp):
    def __init__(self, name):
        super(MaxPoolOp, self).__init__(name)

    def propagateShapes(self):
        self.debugAssert(len(self._inputs) == 1)
        self.debugAssert(len(self._outputs) == 1)

        in_shape = self._inputs[0].shape
        batch_size = in_shape.getDimension(0)
        if self._format == 'NCHW':
            out_chans = in_shape.getDimension(1)
            in_height = in_shape.getDimension(2)
            in_width = in_shape.getDimension(3)
            out_height = in_height // self._strides[2]
            out_width = in_width // self._strides[3]
            out_shape = [batch_size, out_chans, out_height, out_width]
        elif self._format == 'NHWC':
            out_chans = in_shape.getDimension(3)
            in_height = in_shape.getDimension(1)
            in_width = in_shape.getDimension(2)
            out_height = in_height // self._strides[1]
            out_width = in_width // self._strides[2]
            out_shape = [batch_size, out_height, out_width, out_chans]
        else:
            self.notImplemented('Unknown data format: {}'
                                .format(self._format))
        self._outputs[0].shape.mergeShape(out_shape)

    def calcAlgFlops(self):
        self.debugAssert(len(self._inputs) == 1)
        self.debugAssert(len(self._outputs) == 1)

        out_shape = self._outputs[0].shape
        out_size = out_shape.numElements()
        # Use batch size from input
        out_size *= self._inputs[0].shape.getDimension(0).symbol
        out_size /= self._outputs[0].shape.getDimension(0).symbol
        self.debugAssert(self._ksize is not None)
        kernel_area = TensorShape(self._ksize).numElements()
        flops = out_size * kernel_area
        return flops


class MaxPoolGradOp(PoolBaseOp):
    def __init__(self, name):
        super(MaxPoolGradOp, self).__init__(name)

    def propagateShapes(self):
        self.debugAssert(len(self._inputs) == 3)
        self.debugAssert(len(self._outputs) == 1)

        # Simply propagate the input 0 (x) shape to the output, since
        # output is the gradient with respect to x
        self._outputs[0].shape.mergeShape(self._inputs[0].shape)

    def calcAlgFlops(self):
        self.debugAssert(len(self._inputs) == 3)
        self.debugAssert(len(self._outputs) == 1)

        # Use the first input as the out_shape
        out_shape = self._inputs[1].shape
        out_size = out_shape.numElements()
        self.debugAssert(self._ksize is not None)
        kernel_area = TensorShape(self._ksize).numElements()
        flops = out_size * kernel_area
        return flops


class RangeOp(Op):
    def __init__(self, name):
        super(RangeOp, self).__init__(name)

    def propagateShapes(self):
        self.debugAssert(len(self._inputs) == 3)
        self.debugAssert(len(self._outputs) == 1)
        start = self._inputs[0].value
        limit = self._inputs[1].value
        delta = self._inputs[2].value

        if start is None or limit is None or delta is None:
            return

        # Set output shape
        supported_shape_types = (int, sympy.Symbol)
        if self._outputs[0].dtype == DataType.int32:
            start = as_int(start)
            limit = as_int(limit)
            delta = as_int(delta)
            num_elts = (limit - start + delta - 1) // delta
            self._outputs[0].shape.mergeShape([num_elts])
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

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class ReduceOp(Op):
    def __init__(self, name, reduction_op='sum', axes=None):
        super(ReduceOp, self).__init__(name)
        if isinstance(axes, int):
            axes = [axes]
        self._axes = axes
        self._flops_per_element = 1

    def setAxes(self, axes):
        if isinstance(axes, int):
            axes = [axes]
        self._axes = axes

    def propagateShapes(self):
        # First input is always the tensor to be reduced, and the optional
        # second tensor describes the dimensions to be reduced.
        self.debugAssert(len(self._inputs) == 1 or len(self._inputs) == 2)
        self.debugAssert(len(self._outputs) == 1)
        if len(self._inputs) == 2:
            # TODO (Joel): May be too strict
            self.debugAssert(self._axes is None)
            axes = self._inputs[1].value
        else:
            axes = self._axes
        if axes is not None:
            if isinstance(axes, int):
                axes = [axes]
            for idx in range(len(axes)):
                if axes[idx] < 0:
                    axes[idx] += self._inputs[0].shape.rank
            out_shape = []
            for dim_index in range(self._inputs[0].shape.rank):
                if dim_index not in axes:
                    dim = self._inputs[0].shape.getDimension(dim_index)
                    out_shape.append(dim)
            self._outputs[0].shape.mergeShape(out_shape)
            # TODO (Joel): Also calculate value? Depends on the reduction_op!

    def calcAlgFlops(self):
        self.debugAssert(len(self._inputs) == 1 or len(self._inputs) == 2,
            'Reduce {} has too many inputs: {}'
            .format(self.name, [input for input in self._inputs]))
        flops_to_return = self._flops_per_element
        for dim_index in range(self._inputs[0].shape.rank):
            dim = self._inputs[0].shape.getDimension(dim_index)
            flops_to_return *= dim.symbol
        return flops_to_return

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class SelectOp(Op):
    ''' Note similarity to the WhereOp '''
    def __init__(self, name):
        super(SelectOp, self).__init__(name)

    def propagateShapes(self):
        self.debugAssert(len(self._inputs) == 3)
        self.debugAssert(len(self._outputs) == 1)
        # Second and third inputs must be the same size
        in_shape = None
        if not self._inputs[1].shape.isUnknown():
            in_shape = self._inputs[1].shape
        if not self._inputs[2].shape.isUnknown():
            if in_shape is not None:
                self.debugAssert(in_shape == self._inputs[2].shape)
            else:
                in_shape = self._inputs[2].shape
        # Output is same shape as inputs
        if in_shape is not None:
            self._outputs[0].shape.mergeShape(in_shape)

    def calcAlgFlops(self):
        self.debugAssert(len(self._inputs) == 3)
        self.debugAssert(len(self._outputs) == 1)
        # One flop for each of the comparisons (i.e., size of the first
        # input tensor)
        return self._inputs[0].shape.numElements()

    def calcAlgBytes(self):
        # Only access one of the second or third tensors for each element
        # in the input conditioned on the first input tensor.
        input_bytes_accessed = self._inputs[0].size + self._inputs[1].size
        return input_bytes_accessed + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class SoftmaxOp(Op):
    ''' Normalize the input using a soft-max function:
        softmax[i, j] = exp(logits[i, j]) / sum_j(exp(logits[i, j]))
    '''
    def __init__(self, name):
        super(SoftmaxOp, self).__init__(name)

    def propagateShapes(self):
        self.debugAssert(len(self._inputs) == 1)
        self.debugAssert(len(self._outputs) == 1)
        self._outputs[0].shape.mergeShape(self._inputs[0].shape)

    def calcAlgFlops(self):
        self.debugAssert(len(self._inputs) == 1)
        self.debugAssert(len(self._outputs) == 1)
        total_flops = 0
        # Pointwise exponentiate input (assume 1 Flop per exp)
        total_flops += self._inputs[0].shape.numElements()
        # Reduction sum (1 Flop per element)
        total_flops += self._inputs[0].shape.numElements()
        # Pointwise divide (1 Flop per element)
        total_flops += self._inputs[0].shape.numElements()
        return total_flops

    def calcAlgBytes(self):
        # Assume that exponentiated inputs create a tensor in addition
        # to inputs and outputs, and they must be accessed once
        return 2 * self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Assume intermediate exponentiated results contribute to footprint
        return self.bytesAccessInput() + self.bytesAccessOutput()

