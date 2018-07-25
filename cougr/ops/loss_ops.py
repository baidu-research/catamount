from .base_op import Op


class SparseSoftmaxCrossEntropyWithLogitsOp(Op):
    def __init__(self, name):
        super(SparseSoftmaxCrossEntropyWithLogitsOp, self).__init__(name)

    def propagateShapes(self):
        self.debugAssert(len(self._inputs) == 2)
        self.debugAssert(len(self._outputs) == 2)

        # Verify that 0th axis of input tensors match. Use it as output tensor
        # 0 shape. 1th output tensor shape is same is input 0 shape (grads)
        in_0_shape = self._inputs[0].shape
        in_1_shape = self._inputs[1].shape
        in_0_batch_dim = in_0_shape.getDimension(0)
        in_1_batch_dim = in_1_shape.getDimension(0)
        assert in_0_batch_dim == in_1_batch_dim
        self._outputs[0].shape.mergeShape([in_0_batch_dim])
        self._outputs[1].shape.mergeShape(in_0_shape)

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
