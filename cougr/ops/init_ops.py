from .base_op import Op


class IdentityOp(Op):
    def __init__(self, name):
        super(IdentityOp, self).__init__(name)

    def propagateShapes(self):
        # Identity must propagate input size to output size
        assert len(self._inputs) == 1
        assert len(self._outputs) == 1
        assert(self._inputs[0].shape == self._outputs[0].shape)

    def calcAlgFlops(self):
        # IdentityOps have no Flops
        return 0

class RandomInitializerOp(Op):
    def __init__(self, name):
        super(RandomInitializerOp, self).__init__(name)

    def propagateShapes(self):
        # Intializers have no inputs to propagate
        pass

    def calcAlgFlops(self):
        # Intializers have no Flops
        return 0

