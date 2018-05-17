from . import base_op


class AddOp(base_op.Op):
    def __init__(self, name):
        super(AddOp, self).__init__(name)

class MulOp(base_op.Op):
    def __init__(self, name):
        super(MulOp, self).__init__(name)

class MatMulOp(base_op.Op):
    def __init__(self, name):
        super(MatMulOp, self).__init__(name)

class SubOp(base_op.Op):
    def __init__(self, name):
        super(SubOp, self).__init__(name)

