from .base_op import Op


class UnknownOp(Op):
    def __init__(self, name):
        super(UnknownOp, self).__init__(name)
