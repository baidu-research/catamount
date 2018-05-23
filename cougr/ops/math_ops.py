from .base_op import Op


class BasePointwiseOp(Op):
    def __init__(self, name):
        super(BasePointwiseOp, self).__init__(name)
        self._flops_per_element = 1

    def propagateShapes(self):
        # Verify inputs have same shape
        assert(len(self._inputs) == 2)
        in_a_shape = self._inputs[0].shape
        in_b_shape = self._inputs[1].shape
        assert(in_a_shape == in_b_shape)
        assert(len(self._outputs) == 1)
        # Set the output dimensions
        out_shape = self._outputs[0].shape
        for idx, value in enumerate(out_shape.dims):
            out_shape.setDimension(idx, in_a_shape.getDim(idx))

    def flopsPointwise(self):
        assert(len(self._outputs) == 1)
        out_shape = self._outputs[0].shape
        return out_shape.numElements() * self._flops_per_element

    def calcAlgFlops(self):
        return self.flopsPointwise()


class AddOp(BasePointwiseOp):
    def __init__(self, name):
        super(AddOp, self).__init__(name)


class LogicalNotOp(BasePointwiseOp):
    def __init__(self, name):
        super(LogicalNotOp, self).__init__(name)


class MaximumOp(BasePointwiseOp):
    def __init__(self, name):
        super(MaximumOp, self).__init__(name)


class MulOp(BasePointwiseOp):
    def __init__(self, name):
        super(MulOp, self).__init__(name)


class PowOp(BasePointwiseOp):
    def __init__(self, name):
        super(PowOp, self).__init__(name)


class ReluOp(BasePointwiseOp):
    def __init__(self, name):
        super(ReluOp, self).__init__(name)


class RsqrtOp(BasePointwiseOp):
    def __init__(self, name):
        super(RsqrtOp, self).__init__(name)
        # Reciprocal square root is 2 Flops per element
        self._flops_per_element = 2

class SubOp(BasePointwiseOp):
    def __init__(self, name):
        super(SubOp, self).__init__(name)


class Conv2DOp(Op):
    def __init__(self, name):
        super(Conv2DOp, self).__init__(name)

    def calcAlgFlops(self):
        raise NotImplementedError('Conv2D alg Flops not implemented yet!')


class MatMulOp(Op):
    def __init__(self, name):
        super(MatMulOp, self).__init__(name)

    def propagateShapes(self):
        # [_] TODO (Joel): Will need to handle transposes here...
        # Verify that shapes can be correctly resolved
        assert(len(self._inputs) == 2)
        tensor_a = self._inputs[0]
        tensor_b = self._inputs[1]
        inner_dim = tensor_a.shape.getDim(1)
        # [_] TODO (Joel): This assert will be too strict at some point
        assert(inner_dim == tensor_b.shape.getDim(0))
        # Resolve shapes
        tensor_c = self._outputs[0]
        first_dim = tensor_a.shape.getDim(0)
        tensor_c.shape.setDimension(0, first_dim)
        second_dim = tensor_b.shape.getDim(1)
        tensor_c.shape.setDimension(1, second_dim)

    def calcAlgFlops(self):
        # [_] TODO (Joel): Will need to handle transposes here...
        # Get matrix inner dimension
        assert(len(self._inputs) == 2)
        tensor_a = self._inputs[0]
        inner_dim = tensor_a.shape.getDim(1)
        # [_] TODO (Joel): HACK!!!!: FIX ME!!! (Create tensor dimension in
        #                  the tensor_shape code, and add a check... either
        #                  both int and equal, or symbols)
        import sympy
        assert(type(inner_dim) == sympy.Symbol or \
               type(self._inputs[1].shape.getDim(0)) == sympy.Symbol or \
               inner_dim == self._inputs[1].shape.getDim(0))
        # Get output number of elements
        assert(len(self._outputs) == 1)
        out_shape = self._outputs[0].shape
        out_elts = out_shape.numElements()
        return (2 * inner_dim * out_elts)


class ReduceOp(Op):
    def __init__(self, name):
        super(ReduceOp, self).__init__(name)
        print('WARN: ReduceOp should specify reduction type?')

    def calcAlgFlops(self):
        raise NotImplementedError('ReduceOp alg Flops not implemented yet!')


