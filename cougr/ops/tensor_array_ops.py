from .base_op import Op


class TensorArrayOp(Op):
    def __init__(self, name):
        super(TensorArrayOp, self).__init__(name)

    def propagateShapes(self):
        # TODO (Joel): Propagate shapes for TensorArrayOp
        pass

    def calcAlgFlops(self):
        # TensorArrayOps only allow creation of tensor arrays
        return 0
