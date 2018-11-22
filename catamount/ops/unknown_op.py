from .base_op import Op


class UnknownOp(Op):
    _warned_once = False

    def __init__(self, name):
        super(UnknownOp, self).__init__(name)

    def checkAndWarn(self):
        if not UnknownOp._warned_once:
            print('WARN: Graph contains unknown ops. Assuming 0 Flops, ' \
                  '0 bytes accessed, 0 memory footprint for all unknown!')
            UnknownOp._warned_once = True

    def propagateShapes(self, make_symbolic=False):
        self.checkAndWarn()
        # Here, ignore make_symbolic: UnknownOps do not propagate input shapes
        # to output shapes, so the outputs should not need to be made symbolic

    def calcAlgFlops(self):
        self.checkAndWarn()
        return 0

    def calcAlgBytes(self):
        self.checkAndWarn()
        return 0

    def calcAlgFootprint(self):
        self.checkAndWarn()
        return 0
