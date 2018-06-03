from .base_op import Op


class UnknownOp(Op):
    _warned_once = False

    def __init__(self, name):
        super(UnknownOp, self).__init__(name)

    def propagateShapes(self):
        if not UnknownOp._warned_once:
            print('WARN: Graph contains unknown ops. ' \
                  'Not sure how to propagate shapes!')
            UnknownOp._warned_once = True

    def calcAlgFlops(self):
        if not UnknownOp._warned_once:
            print('WARN: Graph contains unknown ops. ' \
                  'Assuming 0 Flops for all unknown!')
            UnknownOp._warned_once = True
        return 0
