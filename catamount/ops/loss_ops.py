from .base_op import Op


class InTopKOp(Op):
    def __init__(self, name):
        super(InTopKOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(len(self._inputs) == 3)
        self.debugAssert(len(self._outputs) == 1)
        pred_shape = self._inputs[0].shape
        pred_batch = pred_shape.getDimension(0)
        target_shape = self._inputs[1].shape
        target_batch = target_shape.getDimension(0)
        self.debugAssert(pred_batch.symbol - target_batch.symbol == 0)
        self._outputs[0].mergeShape(target_shape,
                                    make_symbolic=make_symbolic)

    def calcAlgFlops(self):
        self.debugAssert(len(self._inputs) == 3)
        self.debugAssert(len(self._outputs) == 1)
        # TODO (Joel): Maybe count comparisons for InTopK functionality
        return 0

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class L2LossOp(Op):
    ''' Computes half the L2 norm of a tensor without the sqrt:
        output = sum(t ** 2) / 2
    '''
    def __init__(self, name):
        super(L2LossOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(len(self._inputs) == 1)
        self.debugAssert(len(self._outputs) == 1)
        # Output shape should be a scalar value. No need for make_symbolic
        # since shape cannot be symbolic
        self._outputs[0].mergeShape([])

    def calcAlgFlops(self):
        self.debugAssert(len(self._inputs) == 1)
        self.debugAssert(len(self._outputs) == 1)
        # 2 Flops per element (square and accumulate) and a final divide
        total_flops = 2 * self._outputs[0].shape.numElements() + 1
        return total_flops

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class SparseSoftmaxCrossEntropyWithLogitsOp(Op):
    def __init__(self, name):
        super(SparseSoftmaxCrossEntropyWithLogitsOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(len(self._inputs) == 2)
        self.debugAssert(len(self._outputs) == 2)

        # Verify that 0th axis of input tensors match. Use it as output tensor
        # 0 shape. 1th output tensor shape is same is input 0 shape (grads)
        in_0_shape = self._inputs[0].shape
        in_1_shape = self._inputs[1].shape
        in_0_batch_dim = in_0_shape.getDimension(0)
        in_1_batch_dim = in_1_shape.getDimension(0)
        self.debugAssert(in_0_batch_dim == in_1_batch_dim)
        self._outputs[0].mergeShape([in_0_batch_dim],
                                    make_symbolic=make_symbolic)
        self._outputs[1].mergeShape(in_0_shape,
                                    make_symbolic=make_symbolic)

    def calcAlgFlops(self):
        self.debugAssert(len(self._inputs) == 2)
        self.debugAssert(len(self._outputs) == 2)

        in_shape = self._inputs[0].shape

        #  1) Point-wise exponentiation of input tensor
        pw_ops = in_shape.numElements()
        #  2) Reduction sum across input (last dimension of input tensor)
        red_ops = 1
        for dim_index in range(self._inputs[0].shape.rank):
            dim = self._inputs[0].shape.getDimension(dim_index)
            red_ops *= dim.symbol
        #  3) Point-wise division across full input tensor again (= pw_ops)
        return 2 * pw_ops + red_ops

    def calcAlgBytes(self):
        # TODO (Joel): It might be safe to assume that this op creates an
        # intermediate tensor as part of the memory accesses (reduced values)
        # If necessary, add at a later time.
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()

