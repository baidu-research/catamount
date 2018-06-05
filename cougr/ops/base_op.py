from ..tensors.tensor import *



class Op:
    def __init__(self, name):
        self._name = name
        self._inputs = []
        self._outputs = []
        self._parent = None

    def isControlOp(self):
        return False

    def setParent(self, parent):
        self._parent = parent

    def addInput(self, tensor):
        assert(isinstance(tensor, Tensor))
        self._inputs.append(tensor)

    def addOutput(self, tensor):
        assert(isinstance(tensor, Tensor))
        self._outputs.append(tensor)
        tensor.setProducer(self)

    def canVisit(self, visited_ops):
        ''' Whether this op can be visited given the previous ops that
            have been visited according to the input set visited_ops.
            By default, most ops require that all producer tensors are
            ready before they can be performed. Other ops must override
            this function to get different functionality.
            Args:
                visited_ops: A set of ops that have been previously
                             visited in the graph
        '''
        for in_tensor in self._inputs:
            if in_tensor.producer not in visited_ops:
                return False
        return True

    def propagateShapes(self):
        print('Crashing in op {} propagateShapes...'.format(self._name))
        for idx, input in enumerate(self._inputs):
            print('In tensor[{}] name {}, shape {}, value {}'.format(idx,
                  input.name, input.shape, input.value))
        for idx, output in enumerate(self._outputs):
            print('Out tensor[{}] name {}, shape {}, value {}'.format(idx,
                  output.name, output.shape, output.value))
        raise NotImplementedError('Op propagateShapes not implemented!',
                                  type(self))

    def calcAlgFlops(self):
        raise NotImplementedError('Op calcAlgFlops not implemented!',
                                  type(self))

    def calcAlgBytes(self):
        raise NotImplementedError('Op calcAlgBytes not implemented!',
                                  type(self))

    def calcAlgFootprint(self):
        # NOTE: Maybe take argument for training vs. inference (to decide
        # whether or not to save activations, respectively)
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

    @property
    def parent(self):
        ''' For scoping and hierarchical graph traversals, keep track
            of the parent for this op. Hierarchical traversals will
            traverse all ops regardless of parents, while flat traversals
            will traverse all ops including those who are not direct
            children of an op.
        '''
        return self._parent
