from . import base_op


class VariableOp(base_op.Op):
    def __init__(self, name):
        super(VariableOp, self).__init__(name)

class AssignOp(base_op.Op):
    def __init__(self, name):
        super(AssignOp, self).__init__(name)

