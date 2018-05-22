from .base_op import Op


class IdentityOp(Op):
    def __init__(self, name):
        super(IdentityOp, self).__init__(name)

    def calcAlgFlops(self):
        # IdentityOps have no Flops
        return 0

class RandomInitializerOp(Op):
    def __init__(self, name):
        super(RandomInitializerOp, self).__init__(name)

    def calcAlgFlops(self):
        # Intializers have no Flops
        return 0

