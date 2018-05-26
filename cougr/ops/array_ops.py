from .base_op import Op


class ConcatOp(Op):
    def __init__(self, name, axis=0):
        super(ConcatOp, self).__init__(name)

    def propagateShapes(self):
        # [_] TODO: propagate shapes
        print('ConcatOp: {}, in_a: {}, in_b: {}'
              .format(self._name, self._inputs[0], self._inputs[1]))
        raise NotImplementedError('Must implement ConcatOp propagateShapes')

    def calcAlgFlops(self):
        # ConcatOps have no Flops
        return 0

class ReshapeOp(Op):
    def __init__(self, name):
        super(ReshapeOp, self).__init__(name)

    def calcAlgFlops(self):
        # ReshapeOps have no Flops
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

    def calcAlgFlops(self):
        # ShapeOps have no Flops
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
        # Assume StridedSliceOps have fully specified shapes
        pass

    def calcAlgFlops(self):
        # StridedSliceOps have no Flops
        return 0

class TransposeOp(Op):
    def __init__(self, name):
        super(TransposeOp, self).__init__(name)

    def calcAlgFlops(self):
        # TransposeOps have no Flops
        return 0

