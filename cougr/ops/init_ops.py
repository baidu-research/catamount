from . import base_op


class IdentityOp(base_op.Op):
    def __init__(self, name):
        super(IdentityOp, self).__init__(name)

class RandomInitializerOp(base_op.Op):
    def __init__(self, name):
        super(RandomInitializerOp, self).__init__(name)

