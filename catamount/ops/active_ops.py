from .base_op import Op


class FusedBatchNormBaseOp(Op):
    def __init__(self, name):
        super(FusedBatchNormBaseOp, self).__init__(name)
        self._format = None

    def setDataFormat(self, format):
        if format != 'NCHW' and format != 'NHWC':
            self.notImplemented('Unknown data format: {}'.format(format))
        self._format = format

    def calcAlgBytes(self):
        # TODO (Joel): This might be optimistic, given that these fused ops
        # have intermediate results that are reused.
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class FusedBatchNormOp(FusedBatchNormBaseOp):
    def __init__(self, name):
        super(FusedBatchNormOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(len(self._inputs) == 5)
        self.debugAssert(len(self._outputs) == 5)
        # Input 0 is the batch values and should propagate to output 0
        self._outputs[0].mergeShape(self._inputs[0].shape,
                                    make_symbolic=make_symbolic)
        # Input 1 is the scale factor and input 2 is the offset, which
        # must have the same dimension
        scale_shape = self._inputs[1].shape
        offset_shape = self._inputs[2].shape
        # These shapes should also be equal to the number of input channels
        self.debugAssert(scale_shape == offset_shape)
        # TODO (Joel): Validate shape of inputs 3 and 4, which represent
        # population mean and variance for inference.
        # Propagate scale/offset shape to outputs 1 through 4, which
        # represent the batch mean and variance for running values, and
        # batch mean and variance for the gradient
        for idx in range(1, 5):
            self._outputs[idx].mergeShape(scale_shape,
                                          make_symbolic=make_symbolic)

    def calcAlgFlops(self):
        self.debugAssert(len(self._inputs) == 5)
        self.debugAssert(len(self._outputs) == 5)

        input_shape = self._inputs[0].shape
        input_size = input_shape.numElements()
        if self._format == 'NCHW':
            num_channels = input_shape.getDimension(1).symbol
        elif self._format == 'NHWC':
            num_channels = input_shape.getDimension(3).symbol
        else:
            self.notImplemented('Unknown data format: {}'
                                .format(self._format))
        # Flop counts include:
        # 1) Mean: Reduction sum over elements in each channel
        #    and pointwise divide sums by counts
        flops = input_size + num_channels
        # 2) Center data: Pointwise subtraction
        flops += input_size
        # 3) Variance: Reduction sum over squared elements in each
        #    channel and pointwise divide and square root sums
        flops += 2 * input_size + 2 * num_channels
        # 4) Normed data: Pointwise scale centered data
        flops += input_size
        # 5) Offset norms (for activations): Pointwise add beta to normed data
        flops += input_size
        return flops


class FusedBatchNormGradOp(FusedBatchNormBaseOp):
    def __init__(self, name):
        super(FusedBatchNormGradOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(len(self._inputs) == 5)
        self.debugAssert(len(self._outputs) == 5)
        # Input 0 is the gradient values. Should match the input 1 values,
        # which were the original outputs, and should propagate to output 0,
        # which is the gradient with respect to original input x
        grad_shape = self._inputs[0].shape
        x_shape = self._inputs[1].shape
        self.debugAssert(grad_shape == x_shape)
        self._outputs[0].mergeShape(x_shape,
                                    make_symbolic=make_symbolic)
        # Input 2 is the input scaling factor, which should propagate shape
        # to outputs 1 and 2, the scale and offset gradients
        self._outputs[1].mergeShape(self._inputs[2].shape,
                                    make_symbolic=make_symbolic)
        self._outputs[2].mergeShape(self._inputs[2].shape,
                                    make_symbolic=make_symbolic)
        # TODO (Joel): May need to check shape of inputs 3, 4 for training

    def calcAlgFlops(self):
        self.debugAssert(len(self._inputs) == 5)
        self.debugAssert(len(self._outputs) == 5)

        in_0_shape = self._inputs[0].shape
        in_1_shape = self._inputs[1].shape
        self.debugAssert(in_0_shape == in_1_shape, 'In shapes mismatch: {} {}'
                         .format(in_0_shape, in_1_shape))
        inputs_size = in_0_shape.numElements()
        if self._format == 'NCHW':
            num_channels = in_0_shape.getDimension(1).symbol
        elif self._format == 'NHWC':
            num_channels = in_0_shape.getDimension(3).symbol
        else:
            self.notImplemented('Unknown data format: {}'
                                .format(self._format))
        # Flop counts include:
        # 1) Differentiate the pointwise scaling (4 in FusedBatchNormOp):
        #    1 pointwise multiply and a reduction sum
        flops = 2 * inputs_size
        # 2) Differentiate the squaring and reduction (3 in FusedBatchNormOp)
        flops += 2 * inputs_size
        # 3) Differentiate mean calculation (1 in FusedBatchNormOp)
        #    1 reduction sum, a vector point-wise multiply, and add
        flops += 2 * inputs_size + num_channels
        return flops
