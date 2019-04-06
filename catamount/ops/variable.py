from .base_op import Op
from catamount.tensors.tensor import DataType


class VariableOp(Op):
    def __init__(self, name):
        super(VariableOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        # Variables have no inputs to propagate
        self.debugAssert(len(self._inputs) == 0)

    def calcModelParameters(self):
        # Variables are model parameters, so their output tensor sizes
        # are the count of parameters in the model
        return self._outputs[0].shape.numElements()

    def calcAlgFlops(self):
        # Variables have no Flops
        return 0

    def calcAlgBytes(self):
        # Variable ops just supply a persistent tensor to other ops for
        # consumption. They are not accessed in this op.
        return 0

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class AssignOp(Op):
    def __init__(self, name):
        super(AssignOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        # Assign must propagate input size to output size
        self.debugAssert(len(self._inputs) == 2)
        self.debugAssert(len(self._outputs) == 1)
        self.debugAssert(self._inputs[0].shape.dims is None or \
                         self._inputs[1].shape.dims is None or \
                         self._inputs[0].shape == self._inputs[1].shape)
        self.debugAssert(self._inputs[0].shape == self._outputs[0].shape)
        self._outputs[0].mergeShape(self._inputs[0].shape,
                                    make_symbolic=make_symbolic)

    def calcAlgFlops(self):
        # Assignments have no Flops
        return 0

    def calcAlgBytes(self):
        if not self._inputs[0].shape.isUnknown():
            input_bytes_accessed = self._inputs[0].size
        else:
            input_pytes_accessed = self._inputs[1].size
        return input_bytes_accessed + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class CastOp(Op):
    def __init__(self, name):
        super(CastOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        # Output is same shape as input, propagate if necessary
        self.debugAssert(len(self._inputs) == 1)
        self.debugAssert(len(self._outputs) == 1)
        self._outputs[0].mergeShape(self._inputs[0].shape,
                                    make_symbolic=make_symbolic)
        if self._inputs[0].value is not None:
            # TODO: Loosen this or clear the values if required
            #       (MAYBE IN setValue?!)
            self.debugAssert(self._outputs[0].shape.isFullyNumeric())
            out_val = DataType.cast(self._inputs[0].value,
                                    self._outputs[0].dtype)
            self._outputs[0].setValue(out_val)

    def calcAlgFlops(self):
        # Assignments have no Flops
        return 0

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()

