from .base_op import Op


class VariableOp(Op):
    def __init__(self, name):
        super(VariableOp, self).__init__(name)

    def calcAlgFlops(self):
        # Variables have no Flops
        return 0

class AssignOp(Op):
    def __init__(self, name):
        super(AssignOp, self).__init__(name)

    def calcAlgFlops(self):
        # Assignments have no Flops
        return 0

class CastOp(Op):
    def __init__(self, name):
        super(CastOp, self).__init__(name)

    def calcAlgFlops(self):
        # Assignments have no Flops
        return 0

