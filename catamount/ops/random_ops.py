from .base_op import Op
from catamount.api import utils


class MultinomialOp(Op):
    ''' Samples from an unnormalized multinomial distribution. First input
        is matrix of dimension [batch_size, num_classes], and sampling
        occurs along num_classes dimension. Second input is the number of
        samples to draw for each batch element.
    '''
    def __init__(self, name):
        super(MultinomialOp, self).__init__(name)
        samps_name = '{}::rand_samps'.format(self.name)
        self._num_samples_symbol = \
            num_samples = utils.getIntSymbolFromString(samps_name)

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(len(self._inputs) == 2)
        self.debugAssert(len(self._outputs) == 1)

        # Cannot propagate shapes if first input shape undefined
        if not self._inputs[0].shape.isFullySymbolic():
            return
        self.debugAssert(self._inputs[0].shape.rank == 2)
        num_samples = self._inputs[1].value
        if num_samples == None:
            num_samples = self._num_samples_symbol
        out_shape = []
        out_shape.append(self._inputs[0].shape.getDimension(0))
        out_shape.append(num_samples)
        self._outputs[0].mergeShape(out_shape,
                                    make_symbolic=make_symbolic)

    def calcAlgFlops(self):
        self.debugAssert(len(self._inputs) == 2)
        self.debugAssert(len(self._outputs) == 1)
        # Steps in multinomial sampling:
        # 1) Draw uniform random sample, "noises", of size
        #        [batch_size, num_samples, num_classes]
        num_samples = self._inputs[1].value
        if num_samples == None:
            num_samples = self._num_samples_symbol
        in_0_shape = self._inputs[0].shape
        full_shape_elts = in_0_shape.numElements() * num_samples
        total_flops = full_shape_elts
        # 2) Calculate scores = logits - log(-log(noises)) with broadcasting
        total_flops += 3 * full_shape_elts
        # 3) Minimum reduction along classes dimension
        total_flops += full_shape_elts
        return total_flops

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        return self.bytesAccessOutput()


class CandidateSamplerOp(Op):
    def __init__(self, name):
        super(CandidateSamplerOp, self).__init__(name)
        # TODO (Joel): Read these from compute graph op attributes
        self.setNumTrue(1)
        self.setNumSampled(None)
        # TODO: Depending on the generator, there should be some small number
        # of Flops per sampled element. Using (incorrect) 1 for now...
        self._flops_per_element = 1
        samps_name = '{}::rand_samps'.format(self.name)
        self._num_samples_symbol = \
            num_samples = utils.getIntSymbolFromString(samps_name)

    def setNumTrue(self, num_true):
        self._num_true = num_true

    def setNumSampled(self, num_sampled):
        self._num_sampled = num_sampled

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(len(self._inputs) == 1)
        # First output (output[1]) is the true expected count and has shape
        # equal to the input tensor unless num_true attribute is changed
        self.debugAssert(len(self._outputs) == 3)

        if self._num_true != 1:
            self.notImplemented('CandidateSamplerOp propagateShapes ' \
                                'num_true != 1')
        self._outputs[1].mergeShape(self._inputs[0].shape,
                                    make_symbolic=make_symbolic)
        num_samples = None
        if self._num_sampled is None:
            num_samples = self._num_samples_symbol
        else:
            self.notImplemented('CandidateSamplerOp: propagateShapes '\
                                'num_sampled != None')
        self._outputs[0].mergeShape([num_samples],
                                    make_symbolic=make_symbolic)
        self._outputs[2].mergeShape([num_samples],
                                    make_symbolic=make_symbolic)

    def calcAlgFlops(self):
        # TODO: This is a very conservative estimate that there is one Flop to
        # generate each of the output sample values
        return self._flops_per_element * self._outputs[1].shape.numElements()

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        return self.bytesAccessOutput()
