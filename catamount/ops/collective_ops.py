import sympy

from .base_op import Op
from ..tensors.tensor_shape import Dimension
from catamount.api import utils


class AllgatherOp(Op):
    def __init__(self, name):
        super(AllgatherOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(len(self._inputs) == 2)
        self.debugAssert(len(self._outputs) == 1)
        # Assume that there are multiple workers contributing to this
        # collective operation and their matrix sizes are the same as first
        # input tensor passed in here. Create a symbol to represent the number
        # of participating workers
        num_workers_str = '{}::num_workers'.format(self.name)
        num_workers_symbol = utils.getIntSymbolFromString(num_workers_str)

        # TODO (Joel): We could take another input tensor to specify the axis
        # on which to concatenate values. For now, axis = 0
        axis = 0
        final_shape = []
        for idx in range(len(self._inputs[0].shape.dims)):
            dim = self._inputs[0].shape.getDimension(idx)
            if idx == axis:
                # Manipulate the dimension make the value None (it is
                # necessarily symbolic), and set the symbol to reflect
                # multiple workers
                dim_val = dim.value
                new_dim = Dimension(None)
                new_symbol = dim.symbol * num_workers_symbol
                new_dim.setSymbolOrName(new_symbol)
                final_shape.append(new_dim)
            else:
                final_shape.append(dim)
        self._outputs[0].mergeShape(final_shape,
                                    make_symbolic=make_symbolic)

    def calcAlgFlops(self):
        # Allgathers are only communication and no Flops
        return 0

    def calcAlgBytes(self):
        # Allgathers read and write the whole size of the output tensor twice:
        # Reads to forward the last chunk to the next neighbor, and writes to
        # get the next chunk from the previous neighbor.
        return 2 * self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class AllreduceOp(Op):
    def __init__(self, name):
        super(AllreduceOp, self).__init__(name)
        num_workers_str = '{}::num_workers'.format(self.name)
        self._workers_symbol = utils.getIntSymbolFromString(num_workers_str)

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(len(self._inputs) == 1)
        self.debugAssert(len(self._outputs) == 1)
        self._outputs[0].mergeShape(self._inputs[0].shape,
                                    make_symbolic=make_symbolic)

    def calcAlgFlops(self):
        # Assume one Flop per data element (this is the minimum)
        return self._inputs[0].shape.numElements()

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Must allocate a second tensor for partial sums (but could be
        # smaller than the input, depending on workers)
        return self.bytesAccessOutput()

