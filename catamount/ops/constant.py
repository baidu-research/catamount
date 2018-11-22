from .base_op import Op


class NoOp(Op):
    ''' NoOps are special Catamount ops that fill in a place for non-functional
        compute graph ops, especially for those imported from different
        frameworks. For instance, Tensorflow Saver ops (Save, Restore) have
        no functional purpose in calculating compute graph outputs, but are
        there to load and save tensors.
    '''
    def __init__(self, name):
        super(NoOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        # NoOps should be ignored when propagating shapes
        pass

    def calcAlgFlops(self):
        # NoOps have no Flops
        return 0

    def outputShapeIllDefined(self):
        # Ignore output shape checks for ops considered "NoOp"
        return False

    def calcAlgBytes(self):
        # Assume NoOps access no memory bytes
        return 0

    def calcAlgFootprint(self):
        # Assume NoOps have no tensors to access
        return 0


class ConstantOp(Op):
    def __init__(self, name):
        super(ConstantOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        # Constants must have outputs fully specified. Nothing to propagate
        pass

    def calcAlgFlops(self):
        # Constants have no Flops
        return 0

    def calcAlgBytes(self):
        return self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Just return the output tensor bytes
        return self.bytesAccessOutput()
