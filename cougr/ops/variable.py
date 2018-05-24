from .base_op import Op


class VariableOp(Op):
    def __init__(self, name):
        super(VariableOp, self).__init__(name)

    def propagateShapes(self):
        # Variables have no inputs to propagate
        # [_] TODO (Joel): We should be able to bind variable dimensions
        # to symbols also
        assert len(self._inputs) == 0

    def calcAlgFlops(self):
        # Variables have no Flops
        return 0

class AssignOp(Op):
    def __init__(self, name):
        super(AssignOp, self).__init__(name)

    def propagateShapes(self):
        # Assign must propagate input size to output size
        assert len(self._inputs) == 2
        assert len(self._outputs) == 1
        assert(self._inputs[0].shape.dims is None or \
               self._inputs[1].shape.dims is None or \
               self._inputs[0].shape == self._inputs[1].shape)
        assert(self._inputs[0].shape == self._outputs[0].shape)

    def calcAlgFlops(self):
        # Assignments have no Flops
        return 0

class CastOp(Op):
    def __init__(self, name):
        super(CastOp, self).__init__(name)

    def calcAlgFlops(self):
        # Assignments have no Flops
        return 0

