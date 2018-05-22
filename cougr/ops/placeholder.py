from .base_op import Op


class PlaceholderOp(Op):
    def __init__(self, name):
        super(PlaceholderOp, self).__init__(name)

    def calcAlgFlops(self):
        # Placeholders have no Flops
        return 0
