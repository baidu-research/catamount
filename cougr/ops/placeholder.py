from . import base_op


class PlaceholderOp(base_op.Op):
    def __init__(self, name):
        super(PlaceholderOp, self).__init__(name)
