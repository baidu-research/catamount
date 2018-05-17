from ..tensors.tensor import *



class Op:
    def __init__(self, name):
        self._name = name
        self._inputs = []
        self._outputs = []

    def addInput(self, tensor):
        assert(isinstance(tensor, Tensor))
        self._inputs.append(tensor)

    def addOutput(self, tensor):
        assert(isinstance(tensor, Tensor))
        self._outputs.append(tensor)

    @property
    def name(self):
        return self._name
