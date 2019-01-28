from .base_op import Op
from .subgraph_op import SubgraphOp
from ..api import utils


class ContextFrame:
    def __init__(self, name):
        self._name = name
        self._enter_ops = {}

    @property
    def name(self):
        return self._name

    def __str__(self):
        to_return = 'ContextFrame(name: {}):'.format(self._name)
        for enter_op in self._enter_ops.values():
            to_return += '\n  Enter: {}'.format(enter_op.name)
        return to_return

    def addEnterOp(self, enter_op):
        self._enter_ops[enter_op.name] = enter_op
        enter_op.setContextFrame(self)


class ControlBlockOp(SubgraphOp):
    ''' A ControlBlockOp designates a subgraph that manages some form of
        dynamic control flow for a compute graph (e.g., if-conditionals or
        while loops). Such ops are actually a collection of ops that perform
        the dynamic control operations. Note: ControlBlockOps can contain
        other ControlBlockOps (nesting).
    '''
    def __init__(self, name, root_op, ops_list, enter_ops, exit_ops):
        super(ControlBlockOp, self).__init__(name, ops_list)
        self.debugAssert(isinstance(root_op, Op))
        # The op that controls the execution of the children ops and
        # designation of the type of the control block
        self._root_op = root_op
        self._enter_ops = enter_ops
        self._exit_ops = exit_ops

    def getFreeSymbols(self):
        to_return = super(ControlBlockOp, self).getFreeSymbols()
        # TODO: Move this symbol to a loop-like op
        loop_iter_name = '{}::iters'.format(self.name)
        loop_iters = utils.getIntSymbolFromString(loop_iter_name)
        to_return.add(loop_iters)
        return to_return

    def calcAlgFlops(self):
        if not isinstance(self._root_op, LoopConditionOp):
            raise NotImplementedError(
                ' {} has unknown _root_op type {}'
                .format(type(self), self.name, type(self._root_op)))

        loop_iter_name = '{}::iters'.format(self.name)
        loop_iters = utils.getIntSymbolFromString(loop_iter_name)
        return loop_iters * super(ControlBlockOp, self).calcAlgFlops()

    def calcAlgBytes(self):
        ''' Calculate the algorithmic memory bytes accessed for the compute
        graph based on the ops that depend on ops in the feed_dict.
        '''
        if not isinstance(self._root_op, LoopConditionOp):
            raise NotImplementedError(
                ' {} has unknown _root_op type {}'
                .format(type(self), self.name, type(self._root_op)))

        # Use a hierarchical traversal and allow parents to count for their
        # children.
        ops_to_execute = self.getTopologicalOpOrder(hierarchical=True)
        alg_bytes_one_iter = 0
        enter_exit_op_bytes = 0
        for op in ops_to_execute:
            assert op.parent == self, \
                'Incorrect parent for op {}: {}'.format(op.name, op.parent)
            if isinstance(op, (EnterOp, ExitOp)):
                enter_exit_op_bytes += op.calcAlgBytes()
            else:
                op_alg_bytes = op.calcAlgBytes()
                # print('Op: {}, alg_bytes: {}'.format(op.name, op_alg_bytes))
                alg_bytes_one_iter += op_alg_bytes

        loop_iter_name = '{}::iters'.format(self.name)
        loop_iters = utils.getIntSymbolFromString(loop_iter_name)
        return loop_iters * alg_bytes_one_iter + enter_exit_op_bytes

    def calcAlgFootprint(self):
        ''' Calculate the algorithmic memory footprint to perform the compute
        graph computation.
        '''
        if not isinstance(self._root_op, LoopConditionOp):
            raise NotImplementedError(
                ' {} has unknown _root_op type {}'
                .format(type(self), self.name, type(self._root_op)))

        # Use a hierarchical traversal and allow parents to count for their
        # children.
        ops_to_execute = self.getTopologicalOpOrder(hierarchical=True)
        alg_foot_one_iter = 0
        enter_exit_op_foot = 0
        for op in ops_to_execute:
            assert op.parent == self, \
                'Incorrect parent for op {}: {}'.format(op.name, op.parent)
            if isinstance(op, (EnterOp, ExitOp)):
                enter_exit_op_foot += op.calcAlgFootprint()
            else:
                op_alg_bytes = op.calcAlgFootprint()
                # print('Op: {}, alg_bytes: {}'.format(op.name, op_alg_bytes))
                alg_foot_one_iter += op_alg_bytes

        loop_iter_name = '{}::iters'.format(self.name)
        loop_iters = utils.getIntSymbolFromString(loop_iter_name)
        return loop_iters * alg_foot_one_iter + enter_exit_op_foot


class EnterOp(Op):
    ''' EnterOp designates the start of a control flow operation that acts
        on the input tensor to the op. The output tensor is just the input
        tensor. However, the output tensor will need to be annotated with
        information about the control flow path it is on. For example, for
        variables used inside dynamic loops, the tensor may need to track
        the dynamic instance ID. MergeOps enforce dynamic instance
        versioning, so EnterOps do no real work.
    '''
    def __init__(self, name):
        super(EnterOp, self).__init__(name)
        self._frame_name = None
        self._context_frame = None

    def setFrameName(self, frame_name):
        self.debugAssert(self._frame_name is None)
        self._frame_name = frame_name

    def getFrameName(self):
        return self._frame_name

    def setContextFrame(self, context_frame):
        self._context_frame = context_frame

    def propagateShapes(self, make_symbolic=False):
        # EnterOps should forward their inputs to their outputs
        self.debugAssert(len(self._inputs) == 1)
        self.debugAssert(len(self._outputs) == 1)
        if not self._inputs[0].shape.isUnknown():
            if self._inputs[0].shape != self._outputs[0].shape:
                self.notImplemented('EnterOp propagateShapes {}'
                                    .format(self._name))
            self._outputs[0].mergeShape(self._inputs[0].shape,
                                        make_symbolic=make_symbolic)
        else:
            self.notImplemented(
                'EnterOp {} propagateShapes unknown input shape'
                .format(self._name))
        if self._inputs[0].value is not None:
            self._outputs[0].setValue(self._inputs[0].value)

    def calcAlgFlops(self):
        # EnterOps perform no calculations
        return 0

    def calcAlgBytes(self):
        # EnterOps only forward the input tensor to the
        # output, but they don't access any tensors
        return 0

    def calcAlgFootprint(self):
        # EnterOps only forward the input tensor to the
        # output, but they don't access any tensors
        return 0


class ExitOp(Op):
    ''' ExitOp designates the end of a control flow operation that acts
        on the input tensor to the op. ExitOps make the input tensor
        available to downstream ops (i.e., outside of the context formed
        by the EnterOp-ExitOp pair).
    '''
    def __init__(self, name):
        super(ExitOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        # ExitOps have no outputs to propagate to
        self.debugAssert(len(self._inputs) == 1, 'Op: {}'.format(self._name))
        self.debugAssert(len(self._outputs) == 1, 'Op: {}'.format(self._name))
        if not self._inputs[0].shape.isUnknown():
            if self._inputs[0].shape != self._outputs[0].shape:
                self.notImplemented('ExitOp propagateShapes {}'
                                    .format(self._name))
            self._outputs[0].mergeShape(self._inputs[0].shape,
                                        make_symbolic=make_symbolic)
        else:
            self.notImplemented(
                'ExitOp {} propagateShapes unknown input shape'
                .format(self._name))

    def calcAlgFlops(self):
        # ExitOps perform no calculations
        return 0

    def calcAlgBytes(self):
        # ExitOps only forward the input tensor to the
        # output, but they don't access any tensors
        return 0

    def calcAlgFootprint(self):
        # ExitOps only forward the input tensor to the
        # output, but they don't access any tensors
        return 0


class LoopConditionOp(Op):
    ''' LoopConditionOp takes a boolean input and passes it out to SwitchOps
        as part of dynamic loops. It is a unique identifier op for loops, so
        it is considered to be a control op.
    '''
    def __init__(self, name):
        super(LoopConditionOp, self).__init__(name)

    def isControlOp(self):
        return True

    def propagateShapes(self, make_symbolic=False):
        # LoopConditionOps forward their input to their output
        # [_] TODO (Joel): If shapes are unspecified, bind them
        self.debugAssert(len(self._inputs) == 1)
        self.debugAssert(self._inputs[0].shape.numElements() == 1)
        for out_tensor in self._outputs:
            self.debugAssert(out_tensor.shape.numElements() == 1)

    def calcAlgFlops(self):
        # LoopConditionOps perform no calculations
        return 0

    def calcAlgBytes(self):
        return self.bytesAccessInput() + self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()


class MergeOp(Op):
    ''' MergeOp forwards the value of the first available tensor to the first
        output and sets the second output equal to the index of the first
        available input.
    '''
    def __init__(self, name):
        super(MergeOp, self).__init__(name)

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
        # Check if any inputs are ready
        ready_in_tensors = set()
        for in_tensor in self._inputs:
            if in_tensor.producer in visited_ops:
                ready_in_tensors.add(in_tensor)
        # If at least one input tensor is ready, then can visit
        return len(ready_in_tensors) > 0

    def propagateShapes(self, make_symbolic=False):
        # MergeOps forward their input to their output for the
        # next iteration of a loop
        self.debugAssert(len(self._inputs) >= 1)
        self.debugAssert(len(self._outputs) == 2)
        # NOTE: Any of the input shapes can be unknown, so find one that is
        # known (if one does not exist, cannot propagate)
        in_shape = None
        in_value = None
        in_index = None
        for idx, in_tensor in enumerate(self._inputs):
            if not in_tensor.shape.isUnknown():
                if in_shape is not None:
                    # Verify that all input tensor can be merged
                    self.debugAssert(
                        in_tensor.shape.canBroadcastTogether(in_shape))
                else:
                    in_shape = in_tensor.shape
            if in_value is None and in_tensor.value is not None:
                in_value = in_tensor.value
                in_index = idx
        if not in_shape.isUnknown():
            if in_shape != self._outputs[0].shape:
                self.notImplemented('MergeOp propagateShapes {}'
                                    .format(self._name))
            self._outputs[0].mergeShape(in_shape,
                                        make_symbolic=make_symbolic)
        else:
            self.notImplemented(
                'MergeOp {} propagateShapes unknown input shape'
                .format(self._name))

        # If any of the inputs is ready, propagate it to the outputs
        if in_value is not None:
            self._outputs[0].setValue(in_value)
            self._outputs[1].setValue(in_index)

    def calcAlgFlops(self):
        # MergeOps perform no calculations
        return 0

    def calcAlgBytes(self):
        # MergeOps only forward the first ready input tensors to the
        # output, but they don't access any tensors
        return 0

    def calcAlgFootprint(self):
        # MergeOps only forward the first ready input tensors to the
        # output, but they don't add any new tensors
        return 0


class NextIterationOp(Op):
    ''' NextIterationOp forwards its input to its output for loops.
    '''
    def __init__(self, name):
        super(NextIterationOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        # NextIterationOps forward their input to their output for the
        # next iteration of a loop
        self.debugAssert(len(self._inputs) == 1)
        self.debugAssert(len(self._outputs) == 1)
        if not self._inputs[0].shape.isUnknown():
            if self._inputs[0].shape != self._outputs[0].shape:
                self.notImplemented('NextIterationOp propagateShapes {}'
                                    .format(self._name))
            self._outputs[0].mergeShape(self._inputs[0].shape,
                                        make_symbolic=make_symbolic)
        else:
            self.notImplemented('NextIterationOp {} propagateShapes unknown ' \
                       'input shape'.format(self._name))

    def calcAlgFlops(self):
        # NextIterationOps perform no calculations
        return 0

    def calcAlgBytes(self):
        # NextIterationOps only copy input tensor to the output, but does
        # not read or write tensors
        return 0

    def calcAlgFootprint(self):
        # NextIterationOps only copy input tensor to the output, but does
        # not read or write tensors
        return 0


class SwitchOp(Op):
    ''' The first input to the SwitchOp is the tensor that should be
        forwarded to one of the outputs. The second input gates whether the
        first input gets forwarded to the first or second output. If the
        second input is true, input goes to the first output, or if the
        second input is false, input goes to the second output.
    '''
    def __init__(self, name):
        super(SwitchOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        # SwitchOps have two inputs and two outputs, and they conditionally
        # propagate the first input either to the first or second output
        # depending on whether the second input is true or false, resp.
        self.debugAssert(len(self._inputs) == 2)
        self.debugAssert(self._inputs[1].shape.isScalar())
        self.debugAssert(len(self._outputs) == 2)
        if self._inputs[0].shape.isUnknown():
            self.notImplemented('Switch propagateShapes unknown input shape')

        if self._inputs[0].shape != self._outputs[0].shape:
            self.notImplemented('SwitchOp propagateShapes output 0')
        self._outputs[0].mergeShape(self._inputs[0].shape,
                                    make_symbolic=make_symbolic)
        if self._inputs[0].value is not None:
            self._outputs[0].setValue(self._inputs[0].value)

        if self._inputs[0].shape != self._outputs[1].shape:
            self.notImplemented('SwitchOp propagateShapes output 1')
        self._outputs[1].mergeShape(self._inputs[0].shape,
                                    make_symbolic=make_symbolic)
        if self._inputs[0].value is not None:
            self._outputs[1].setValue(self._inputs[0].value)

    def calcAlgFlops(self):
        # SwitchOps perform no calculations
        return 0

    def calcAlgBytes(self):
        # SwitchOps only copy the input tensor to the appropriate output,
        # but do not read the inputs or write outputs
        return 0

    def calcAlgFootprint(self):
        # SwitchOps only copy the input tensor to the appropriate output,
        # but do not read the inputs or write outputs
        return 0

