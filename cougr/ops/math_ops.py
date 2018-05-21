from .base_op import Op



class BasePointwiseOp(Op):
    def __init__(self, name):
        super(BasePointwiseOp, self).__init__(name)
        self._flops_per_element = 1

    def flops_pointwise(self, op):
        assert(len(op.outputs) == 1)
        out_shape = op.outputs[0].shape
        return out_shape.numElements() * self._flops_per_element

    def calcAlgFlops(self):
        return self.flops_pointwise(self)


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

    def calcAlgFlops(self):
        # Get matrix inner dimension
        assert(len(self._inputs) == 2)
        tensor_a = self._inputs[0]
        inner_dim = tensor_a.shape.getDim(1)
        assert(inner_dim == self._inputs[1].shape.getDim(0))
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


