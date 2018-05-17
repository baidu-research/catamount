from .base_op import Op


def flops_pointwise(op, flops_per_element=1):
    assert(len(op.outputs) == 1)
    out_shape = op.outputs[0].shape
    return out_shape.numElements() * flops_per_element


class AddOp(Op):
    def __init__(self, name):
        super(AddOp, self).__init__(name)

    def calcAlgFlops(self):
        return flops_pointwise(self)


class MulOp(Op):
    def __init__(self, name):
        super(MulOp, self).__init__(name)

    def calcAlgFlops(self):
        return flops_pointwise(self)


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


class SubOp(Op):
    def __init__(self, name):
        super(SubOp, self).__init__(name)

    def calcAlgFlops(self):
        return flops_pointwise(self)

