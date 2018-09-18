from ..tensors.tensor import *



class Op:
    def __init__(self, name):
        self._name = name
        self._inputs = []
        self._outputs = []
        self._parent = None

    def debugString(self):
        to_return = 'In op {} of type {}:'.format(self._name, type(self))
        for in_tensor in self._inputs:
            to_return += '\n  In:  {}'.format(in_tensor)
        for out_tensor in self._outputs:
            to_return += '\n  Out: {}'.format(out_tensor)
        return to_return

    def notImplemented(self, string):
        # A helper function for printing op details before a crash
        raise NotImplementedError('{}\n{}'.format(string, self.debugString()))

    def debugAssert(self, condition, message=''):
        if not condition:
            raise AssertionError('{}\n{}'.format(message, self.debugString()))

    def isValid(self):
        for in_tensor in self._inputs:
            if not in_tensor.isValid():
                print('WARN: tensor {} not valid for op {}'
                      .format(in_tensor.name, self._name))
                return False
            if self._name not in in_tensor.consumers.keys():
                print('WARN: tensor {} not consumed by op {}'
                      .format(in_tensor.name, self._name))
                return False
        for out_tensor in self._outputs:
            if not out_tensor.isValid():
                print('WARN: tensor {} not valid for op {}'
                      .format(out_tensor.name, self._name))
                return False
            if out_tensor.producer is not self:
                print('WARN: tensor {} not produced by op {}'
                      .format(out_tensor.name, self._name))
                return False
        return True

    def isControlOp(self):
        return False

    def setParent(self, parent):
        self._parent = parent

    def addInput(self, tensor):
        assert(isinstance(tensor, Tensor))
        self._inputs.append(tensor)

    def resetInputs(self):
        # For each input tensor, remove op from consumers list
        for in_tensor in self._inputs:
            in_tensor.removeConsumer(self)
        self._inputs = []

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
        for in_tensor in self.inputs:
            if in_tensor.producer not in visited_ops:
                return False
        return True

    def bindTensorShapeDimension(self, dim_index, dim_name_or_symbol,
                                 make_symbolic=False):
        self.notImplemented('BaseOp bindTensorShapeDim not implemented!\n' \
                            '  Consider only binding Placeholder or Variable')

    def propagateShapes(self, make_symbolic=False):
        self.notImplemented('Op propagateShapes not implemented! {}'
                            .format(type(self)))

    def calcModelParameters(self):
        # Only VariableOps will have parameters (their output tensors), so by
        # default, assume there are no parameters in an Op.
        return 0

    def calcAlgFlops(self):
        self.notImplemented('Op calcAlgFlops not implemented! {}'
                            .format(type(self)))

    def calcAlgBytes(self):
        self.notImplemented('Op calcAlgBytes not implemented! {}'
                            .format(type(self)))

    def calcAlgFootprint(self):
        # NOTE: Maybe take argument for training vs. inference (to decide
        # whether or not to save activations, respectively)
        self.notImplemented('Op calcAlgFootprint not implemented! {}'
                            .format(type(self)))

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

    def outputShapeIllDefined(self):
        for out_tensor in self.outputs:
            if not out_tensor.shape.isFullySymbolic():
                return True
        return False

    def bytesAccessInput(self):
        total_bytes_input = 0
        for in_tensor in self.inputs:
            total_bytes_input += in_tensor.size
        return total_bytes_input

    def bytesAccessOutput(self):
        total_bytes_output = 0
        for out_tensor in self.outputs:
            total_bytes_output += out_tensor.size
        return total_bytes_output
