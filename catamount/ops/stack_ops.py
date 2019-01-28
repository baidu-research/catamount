from .base_op import Op
from catamount.tensors.tensor import Tensor


class TensorStack:
    ''' An object to be shared among Stack ops to push and pop tensor handles
    for dynamic execution.
    '''
    def __init__(self):
        self._tensor_stack = []
        self._parent = None
        self._push = None
        self._pop = None
        # The tensor reference that will be pushed into the stack
        self._reference = None

    def __len__(self):
        return len(self._tensor_stack)

    def isValid(self):
        return self._parent is not None and \
               self._push is not None and \
               self._pop is not None

    def associateStackOp(self, stack_op):
        self._parent = stack_op

    def associatePush(self, push_op):
        self._push = push_op

    def associatePop(self, pop_op):
        self._pop = pop_op

    @property
    def name(self):
        return self._parent.name

    def addReferenceTensor(self, tensor):
        assert self._reference is None
        self._reference = tensor

    def push(self, tensor):
        assert isinstance(tensor, Tensor)
        self._tensor_stack.insert(0, tensor)

    def peek(self):
        if len(self._tensor_stack) == 0:
            return None
        else:
            return self._tensor_stack[0]


class BaseStackOp(Op):
    def __init__(self, name):
        super(BaseStackOp, self).__init__(name)
        # The stack reference to use for pushing and popping
        self._stack = None

    def debugString(self):
        to_return = super(BaseStackOp, self).debugString()
        to_return += '\n  Stack: {}'.format(self._stack.name)
        return to_return

    def setStack(self, stack):
        self.debugAssert(self._stack is None)
        self._stack = stack

    def getStack(self):
        return self._stack

    def calcAlgFlops(self):
        # Stack operations have no Flops
        return 0

    def calcAlgBytes(self):
        # Stack operations do not perform algorithmic activity,
        # so accessed memory is not algorithmic
        return 0

    def calcAlgFootprint(self):
        # Stack operations do not perform algorithmic activity,
        # so accessed memory is not algorithmic
        return 0


class StackOp(BaseStackOp):
    def __init__(self, name):
        super(StackOp, self).__init__(name)
        self._stack = TensorStack()
        self._stack.associateStackOp(self)

    def isValid(self):
        return self._stack.isValid() and super(StackOp, self).isValid()

    def propagateShapes(self, make_symbolic=False):
        # Zero or one inputs. First input is the maximum depth of the stack
        self.debugAssert(len(self._inputs) <= 1)
        self.debugAssert(len(self._outputs) == 1)
        # The output is a resource handle of shape [Dimension(2)]
        self.debugAssert(self._outputs[0].shape.rank == 1 and
                         self._outputs[0].shape.numElements() == 2)


class StackPopOp(BaseStackOp):
    def __init__(self, name):
        super(StackPopOp, self).__init__(name)

    @property
    def inputs(self):
        tensor_inputs = list(super(StackPopOp, self).inputs)
        if self._stack is not None:
            tensor_inputs.append(self._stack._reference)
        return tensor_inputs

    def setStack(self, stack):
        super(StackPopOp, self).setStack(stack)
        self._stack.associatePop(self)

    def canVisit(self, visited_ops):
        self.debugAssert(self._stack._reference is not None)
        stack_push_op = self._stack._reference.producer
        self.debugAssert(stack_push_op == self._stack._push)
        self.debugAssert(isinstance(stack_push_op, StackPushOp))
        if stack_push_op not in visited_ops:
            return False
        return super(StackPopOp, self).canVisit(visited_ops)

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(len(self._inputs) == 1)
        self.debugAssert(len(self._outputs) == 1)

        self.debugAssert(self._stack._reference is not None)
        in_tensor = self._stack._reference
        in_shape = in_tensor.shape
        self._outputs[0].mergeShape(in_shape, make_symbolic=make_symbolic)

        if in_tensor.value is not None:
            self._outputs[0].setValue(in_tensor.value)


class StackPushOp(BaseStackOp):
    def __init__(self, name):
        super(StackPushOp, self).__init__(name)

    def setStack(self, stack):
        super(StackPushOp, self).setStack(stack)
        self._stack.associatePush(self)
        self._stack.addReferenceTensor(self.outputs[0])

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(len(self._inputs) == 2)
        self.debugAssert(len(self._outputs) == 1)

        in_shape = self._inputs[1].shape
        self._outputs[0].mergeShape(in_shape, make_symbolic=make_symbolic)

        if self._inputs[1].value is not None:
            self._outputs[0].setValue(self._inputs[1].value)
