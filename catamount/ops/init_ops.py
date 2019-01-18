from .base_op import Op


class IdentityOp(Op):
    def __init__(self, name):
        super(IdentityOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        # Identity must propagate input size to output size
        self.debugAssert(len(self._inputs) == 1)
        self.debugAssert(len(self._outputs) == 1)
        self.debugAssert(self._inputs[0].shape.isUnknown() or
                         self._inputs[0].shape == self._outputs[0].shape)
        if not self._inputs[0].shape.isUnknown():
            self._outputs[0].mergeShape(self._inputs[0].shape,
                                        make_symbolic=make_symbolic)

        if self._inputs[0].value is not None:
            self._outputs[0].setValue(self._inputs[0].value)

    def calcAlgFlops(self):
        # IdentityOps have no Flops
        return 0

    def calcAlgBytes(self):
        # It is assumed that IdentityOps just pass a reference from the
        # input to the output without copying the data.
        return 0

    def calcAlgFootprint(self):
        # It is assumed that IdentityOps just pass a reference from the
        # input to the output without copying the data.
        return 0


class PreventGradientOp(IdentityOp):
    ''' PreventGradient just performs the same operations as an IdentityOp,
        but it prevents the gradient from backproping through the tensor. It
        is commonly used in auto-generated backprop graphs to verify that
        backprop does not occur through this part of the graph.
        Here, we distinguish it from the IdentityOp as a way to modify its
        behavior as desired.
    '''
    def __init__(self, name):
        super(PreventGradientOp, self).__init__(name)


class RandomInitializerOp(Op):
    def __init__(self, name):
        super(RandomInitializerOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        # Intializers have input[0] as shape to propagate
        self.debugAssert(len(self._inputs) >= 1)
        self.debugAssert(len(self._outputs) == 1)
        if self._inputs[0].value is not None:
            self._outputs[0].mergeShape(self._inputs[0].value,
                                        make_symbolic=make_symbolic)

    def calcAlgFlops(self):
        # Intializers have no Flops
        return 0

    def calcAlgBytes(self):
        return self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class StopGradientOp(IdentityOp):
    ''' StopGradient just performs the same operations as an IdentityOp,
        but it masks any gradient value from backproping through to the
        upstream ops. It is commonly used in auto-generated backprop
        graphs to fix some values in the graph to be constant and keep
        gradients from flowing to them.
        Here, we distinguish it from the IdentityOp as a way to modify its
        behavior as desired.
    '''
    def __init__(self, name):
        super(StopGradientOp, self).__init__(name)
