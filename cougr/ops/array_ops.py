from . import base_op


class ShapeOp(base_op.Op):
    def __init__(self, name):
        super(ShapeOp, self).__init__(name)

    def calcAlgFlops(self):
        # ShapeOps have no Flops
        return 0

class SplitOp(base_op.Op):
    def __init__(self, name):
        super(SplitOp, self).__init__(name)

    def calcAlgFlops(self):
        # SplitOps have no Flops
        return 0

class StackOp(base_op.Op):
    def __init__(self, name):
        super(StackOp, self).__init__(name)

    def calcAlgFlops(self):
        # StackOps have no Flops
        return 0

class StridedSliceOp(base_op.Op):
    def __init__(self, name):
        super(StridedSliceOp, self).__init__(name)

    def calcAlgFlops(self):
        # StridedSliceOps have no Flops
        return 0

