from ..tensors.tensor import *
from ..api import utils


class Op:
    def __init__(self, name):
        self._name = name
        self._inputs = []
        self._outputs = []
        self._parent = None

    def debugString(self):
        to_return = 'Op(name: {}, type {}):'.format(self._name, type(self))
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

    def getFreeSymbols(self):
        to_return = set()
        for out_tens in self._outputs:
            to_return.update(out_tens.getFreeSymbols())
        return to_return

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

    def calcMinimalFootprint(self, feed_dict=None, fetches_dict=None,
                             verbose=False, symbol_subs=None):
        # NOTE: Maybe take argument for training vs. inference (to decide
        # whether or not to save activations, respectively)
        raise RuntimeError('Op type {} does not support calcMinimalFootprint,'\
                           ' which is a subgraph function!'.format(type(self)))

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

    def calcMinimalFootprintSub(self, max_footprint, curr_footprint,
                                tensors_to_consume, visited_ops,
                                verbose=False, symbol_subs=None):
        # print('  Traversing {}, starting foot {}, max foot {}'
        #       .format(self.name, curr_footprint, max_footprint))
        if self.calcAlgFootprint() == 0:
            visited_ops.add(self)
            return max_footprint, curr_footprint
        # Calculate the size of additional tensors that must be allocated
        # to execute this op
        my_added_footprint = 0
        for out_tensor in self.outputs:
            self.debugAssert(out_tensor not in tensors_to_consume.keys())
            tensors_to_consume[out_tensor] = out_tensor
            my_added_footprint += out_tensor.size

        # The maximum footprint grows if the current footprint plus the
        # additional footprint for this op exceed the maximum
        my_curr_footprint = curr_footprint + my_added_footprint
        # print('    My added foot {} -> {}'.format(my_added_footprint,
        #       my_curr_footprint))
        max_footprint = utils.getSymbolicMaximum(my_curr_footprint,
                                                 max_footprint,
                                                 symbol_subs)

        # Execute the op: This op has now been visited
        visited_ops.add(self)

        # Now remove any input tensors that can be freed
        for in_tensor in self.inputs:
            tensor_can_be_freed = True
            for consumer in in_tensor.consumers.values():
                if consumer not in visited_ops:
                    tensor_can_be_freed = False
                    break
            if tensor_can_be_freed:
                tensors_to_consume.pop(in_tensor, None)
                my_curr_footprint -= in_tensor.size
        if symbol_subs is not None and \
           isinstance(my_curr_footprint, sympy.Expr):
            my_curr_footprint = my_curr_footprint.subs(symbol_subs)
        if verbose:
            if isinstance(my_curr_footprint, sympy.Expr):
                my_int_curr_foot = my_curr_footprint.subs(symbol_subs)
            else:
                my_int_curr_foot = my_curr_footprint
            print('    FOOT: {} {} {}'.format(self.name, max_footprint,
                                              my_int_curr_foot))
        return max_footprint, my_curr_footprint

