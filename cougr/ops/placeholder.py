from . import base_op


class PlaceholderOp(base_op.Op):
    def __init__(self, name):
        super(PlaceholderOp, self).__init__(name)

    def calcAlgFlops(self):
        # Placeholders have no Flops
        return 0
