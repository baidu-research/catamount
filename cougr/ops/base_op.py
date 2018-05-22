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
        tensor.setProducer(self)

    def propagateShapes(self):
        print('Crashing in op {} propagateShapes...'.format(self._name))
        for idx, input in enumerate(self._inputs):
            print('In tensor[{}] name {}, shape {}'.format(idx,
                  input.name, input.shape)) 
        for idx, output in enumerate(self._outputs):
            print('Out tensor[{}] name {}, shape {}'.format(idx,
                  output.name, output.shape)) 
        raise NotImplementedError('Op propagateShapes not implemented!',
                                  type(self))

    def calcAlgFlops(self):
        raise NotImplementedError('Op calcAlgFlops not implemented!',
                                  type(self))

    def calcAlgBytes(self):
        raise NotImplementedError('Op calcAlgBytes not implemented!',
                                  type(self))

    def calcAlgFootprint(self):
        raise NotImplementedError('Op calcAlgFootprint not implemented!',
                                  type(self))

    @property
    def name(self):
        return self._name

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs
