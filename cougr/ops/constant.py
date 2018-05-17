from . import base_op


class ConstOp(base_op.Op):
    def __init__(self, name):
        super(ConstOp, self).__init__(name)

