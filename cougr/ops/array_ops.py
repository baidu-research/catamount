from .base_op import Op


class ReshapeOp(Op):
    def __init__(self, name):
        super(ReshapeOp, self).__init__(name)

    def calcAlgFlops(self):
        # ReshapeOps have no Flops
        return 0

class ShapeOp(Op):
    def __init__(self, name):
        super(ShapeOp, self).__init__(name)

    def calcAlgFlops(self):
        # ShapeOps have no Flops
        return 0

class SplitOp(Op):
    def __init__(self, name):
        super(SplitOp, self).__init__(name)

    def calcAlgFlops(self):
        # SplitOps have no Flops
        return 0

class StackOp(Op):
    def __init__(self, name):
        super(StackOp, self).__init__(name)

    def calcAlgFlops(self):
        # StackOps have no Flops
        return 0

class StridedSliceOp(Op):
    def __init__(self, name):
        super(StridedSliceOp, self).__init__(name)

    def calcAlgFlops(self):
        # StridedSliceOps have no Flops
        return 0

class TransposeOp(Op):
    def __init__(self, name):
        super(TransposeOp, self).__init__(name)

    def calcAlgFlops(self):
        # TransposeOps have no Flops
        return 0

