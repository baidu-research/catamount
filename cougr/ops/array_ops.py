from . import base_op


class ShapeOp(base_op.Op):
    def __init__(self, name):
        super(ShapeOp, self).__init__(name)

class SplitOp(base_op.Op):
    def __init__(self, name):
        super(SplitOp, self).__init__(name)

class StackOp(base_op.Op):
    def __init__(self, name):
        super(StackOp, self).__init__(name)

class StridedSliceOp(base_op.Op):
    def __init__(self, name):
        super(StridedSliceOp, self).__init__(name)
