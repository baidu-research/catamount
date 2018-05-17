from . import base_op


class VariableOp(base_op.Op):
    def __init__(self, name):
        super(VariableOp, self).__init__(name)

    def calcAlgFlops(self):
        # Variables have no Flops
        return 0

class AssignOp(base_op.Op):
    def __init__(self, name):
        super(AssignOp, self).__init__(name)

    def calcAlgFlops(self):
        # Assignments have no Flops
        return 0

