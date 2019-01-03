from .base_op import Op


class ApplyGradientDescentOp(Op):
    ''' Gradient Descent optimizer apply function:
        Update the variable (input[0]) by subtracting (learning_rate
        (input[1]) * gradient (input[2])) from it.
    '''
    def __init__(self, name):
        super(ApplyGradientDescentOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(len(self._inputs) == 3)
        self.debugAssert(self._inputs[0].shape == self._inputs[2].shape)
        self.debugAssert(len(self._outputs) == 1)
        out_shape = self._inputs[0].shape
        self._outputs[0].mergeShape(out_shape,
                                    make_symbolic=make_symbolic)
        if self._inputs[0].value is not None and \
           self._inputs[1].value is not None and \
           self._inputs[2].value is not None:
            self.notImplemented('ApplyGradientDescentOp value prop!')

    def calcAlgFlops(self):
        self.debugAssert(len(self._inputs) == 3)
        self.debugAssert(self._inputs[0].shape == self._inputs[2].shape)
        self.debugAssert(len(self._outputs) == 1)
        # Updates the input tensor in-place, so the output is unused
        self.debugAssert(len(self._outputs[0].consumers) == 0)
        # Gradient pointwise multiplication with learning rate
        pw_mul_flops = self._inputs[2].shape.numElements()
        # Pointwise subtract the weighted gradient from weights
        pw_sub_flops = self._inputs[0].shape.numElements()
        total_flops = pw_mul_flops + pw_sub_flops
        return total_flops

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # NOTE: The output tensor is the variable input[0], which is
        # a persistent tensor. There is no extra footprint here.
        return 0


class ApplyMomentumOp(Op):
    ''' Nesterov Momentum optimizer apply function:
        Update the variables, weights (input[0]) and momentum accumulator
        (input[1]) with learning_rate (input[2]), gradient (input[3]), and
        momentum_decay (input[4]):

        accumulator = accumulator * momentum_decay + gradient
        weights -= learning_rate * accumulator
    '''
    def __init__(self, name):
        super(ApplyMomentumOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(len(self._inputs) == 5)
        self.debugAssert(self._inputs[0].shape == self._inputs[1].shape)
        self.debugAssert(self._inputs[0].shape == self._inputs[3].shape)
        self.debugAssert(len(self._outputs) == 1)
        out_shape = self._inputs[0].shape
        self._outputs[0].mergeShape(out_shape,
                                    make_symbolic=make_symbolic)
        if self._inputs[0].value is not None and \
           self._inputs[1].value is not None and \
           self._inputs[2].value is not None and \
           self._inputs[3].value is not None and \
           self._inputs[4].value is not None:
            self.notImplemented('ApplyMomentumOp value prop!')

    def calcAlgFlops(self):
        self.debugAssert(len(self._inputs) == 5)
        self.debugAssert(self._inputs[0].shape == self._inputs[1].shape)
        self.debugAssert(self._inputs[0].shape == self._inputs[3].shape)
        self.debugAssert(len(self._outputs) == 1)
        # Updates the input tensor in-place, so the output is unused
        self.debugAssert(len(self._outputs[0].consumers) == 0)
        # Momentum pointwise multiplication with decay
        pw_mul_accum_flops = self._inputs[1].shape.numElements()
        # Momentum pointwise add gradient
        pw_add_accum_flops = self._inputs[1].shape.numElements()
        # Gradient pointwise multiplication with learning rate
        pw_mul_grad_flops = self._inputs[3].shape.numElements()
        # Pointwise subtract the weighted gradient from weights
        pw_sub_flops = self._inputs[0].shape.numElements()
        total_flops = pw_mul_accum_flops + pw_add_accum_flops + \
                      pw_mul_grad_flops + pw_sub_flops
        return total_flops

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # NOTE: The output tensors are the variables, input[0] and input[1],
        # which are persistent tensors. There is no extra footprint here.
        return 0
