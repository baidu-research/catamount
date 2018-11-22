from .base_op import Op


class TensorArrayOp(Op):
    def __init__(self, name):
        super(TensorArrayOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        # TODO (Joel): Propagate shapes for TensorArrayOp
        pass

    def calcAlgFlops(self):
        # TensorArrayOps only allow creation of tensor arrays
        return 0

    def calcAlgBytes(self):
        # TODO (Joel): Currently, we do not know how tensor array
        # ops function. Just return 0 and fix later!
        return 0

    def calcAlgFootprint(self):
        # TODO (Joel): Return 0 memory bytes footprint for now and
        # implement this later!
        return 0
