from .base_op import Op
from cougr.api import utils


class MultinomialOp(Op):
    ''' Samples from an unnormalized multinomial distribution. First input
        is matrix of dimension [batch_size, num_classes], and sampling
        occurs along num_classes dimension. Second input is the number of
        samples to draw for each batch element.
    '''
    def __init__(self, name):
        super(MultinomialOp, self).__init__(name)

    def propagateShapes(self):
        self.debugAssert(len(self._inputs) == 2)
        self.debugAssert(len(self._outputs) == 1)

        # Cannot propagate shapes if first input shape undefined
        if not self._inputs[0].shape.isFullySymbolic():
            return
        self.debugAssert(self._inputs[0].shape.rank == 2)
        num_samples = self._inputs[1].value
        if num_samples == None:
            samps_name = '{}::rand_samps'.format(self.name)
            num_samples = utils.getIntSymbolFromString(samps_name)
        out_shape = []
        out_shape.append(self._inputs[0].shape.getDimension(0))
        out_shape.append(num_samples)
        self._outputs[0].shape.mergeShape(out_shape)

    def calcAlgFlops(self):
        self.debugAssert(len(self._inputs) == 2)
        self.debugAssert(len(self._outputs) == 1)
        # Steps in multinomial sampling:
        # 1) Draw uniform random sample, "noises", of size
        #        [batch_size, num_samples, num_classes]
        num_samples = self._inputs[1].value
        if num_samples == None:
            samps_name = '{}::rand_samps'.format(self.name)
            num_samples = utils.getIntSymbolFromString(samps_name)
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
