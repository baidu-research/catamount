from .base_op import Op


class EnterOp(Op):
    ''' EnterOp designates the start of a control flow operation that acts
        on the input tensor to the op. The output tensor is just the input
        tensor, but the output tensor may need to be annotated with
        information about the control flow path it is on. For example, for
        variables used inside dynamic loops, the tensor may need to track
        the dynamic instance ID.
    '''
    def __init__(self, name):
        super(EnterOp, self).__init__(name)

    def propagateShapes(self):
        # EnterOps should forward their inputs to their outputs
        assert len(self._inputs) == 1
        assert len(self._outputs) == 1
        if not self._inputs[0].shape.isUnknown():
            if self._inputs[0].shape != self._outputs[0].shape:
                raise NotImplementedError('EnterOp propagateShapes {}'
                                          .format(self._name))
            self._outputs[0].shape.mergeShape(self._inputs[0].shape)
        else:
            fail_str = 'EnterOp {} propagateShapes unknown input shape' \
                       .format(self._name)
            raise NotImplementedError(fail_str)

    def calcAlgFlops(self):
        # EnterOps perform no calculations
        return 0


class ExitOp(Op):
    ''' ExitOp designates the end of a control flow operation that acts
        on the input tensor to the op. ExitOps make the input tensor
        available to downstream ops (i.e., outside of the context formed
        by the EnterOp-ExitOp pair).
    '''
    def __init__(self, name):
        super(ExitOp, self).__init__(name)

    def propagateShapes(self):
        # ExitOps have no outputs to propagate to
        assert len(self._inputs) == 1, 'Op: {}'.format(self._name)
        assert len(self._outputs) == 1, 'Op: {}'.format(self._name)
        if not self._inputs[0].shape.isUnknown():
            if self._inputs[0].shape != self._outputs[0].shape:
                raise NotImplementedError('ExitOp propagateShapes {}'
                                          .format(self._name))
            self._outputs[0].shape.mergeShape(self._inputs[0].shape)
        else:
            fail_str = 'ExitOp {} propagateShapes unknown input shape' \
                       .format(self._name)
            raise NotImplementedError(fail_str)

    def calcAlgFlops(self):
        # ExitOps perform no calculations
        return 0


class LoopConditionOp(Op):
    ''' LoopConditionOp takes a boolean input and passes it out to SwitchOps
        as part of dynamic loops.
    '''
    def __init__(self, name):
        super(LoopConditionOp, self).__init__(name)

    def propagateShapes(self):
        # LoopConditionOps forward their input to their output
        # [_] TODO (Joel): If shapes are unspecified, bind them
        assert len(self._inputs) == 1
        assert self._inputs[0].shape.numElements() == 1
        for out_tensor in self._outputs:
            assert out_tensor.shape.numElements() == 1

    def calcAlgFlops(self):
        # LoopConditionOps perform no calculations
        return 0


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
        # If (at least?) one input tensor is ready, then can visit
        # [_] TODO (Joel): May need to loosen this restriction
        assert len(ready_in_tensors) <= 1
        return len(ready_in_tensors) == 1

    def propagateShapes(self):
        # MergeOps forward their input to their output for the
        # next iteration of a loop
        assert len(self._inputs) >= 1
        assert len(self._outputs) == 2
        # NOTE: Any of the input shapes can be unknown, so find one that is
        # known (if one does not exist, cannot propagate)
        in_shape = None
        for in_tensor in self._inputs:
            if not in_tensor.shape.isUnknown():
                if in_shape is not None:
                    # Verify that all input tensor can be merged
                    assert in_tensor.shape.canBroadcastTogether(in_shape)
                else:
                    in_shape = in_tensor.shape
        if not in_shape.isUnknown():
            if in_shape != self._outputs[0].shape:
                raise NotImplementedError('MergeOp propagateShapes {}'
                                          .format(self._name))
            self._outputs[0].shape.mergeShape(in_shape)
        else:
            fail_str = 'MergeOp {} propagateShapes unknown input shape' \
                       .format(self._name)
            raise NotImplementedError(fail_str)

    def calcAlgFlops(self):
        # MergeOps perform no calculations
        return 0


class NextIterationOp(Op):
    ''' NextIterationOp forwards its input to its output for loops.
    '''
    def __init__(self, name):
        super(NextIterationOp, self).__init__(name)

    def propagateShapes(self):
        # NextIterationOps forward their input to their output for the
        # next iteration of a loop
        assert len(self._inputs) == 1
        assert len(self._outputs) == 1
        if not self._inputs[0].shape.isUnknown():
            if self._inputs[0].shape != self._outputs[0].shape:
                raise NotImplementedError('NextIterationOp propagateShapes {}'
                                          .format(self._name))
            self._outputs[0].shape.mergeShape(self._inputs[0].shape)
        else:
            fail_str = 'NextIterationOp {} propagateShapes unknown input '\
                       ' shape'.format(self._name)
            raise NotImplementedError(fail_str)

    def calcAlgFlops(self):
        # NextIterationOps perform no calculations
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

    def propagateShapes(self):
        # SwitchOps have two inputs and two outputs, and they conditionally
        # propagate the first input either to the first or second output
        # depending on whether the second input is true or false, resp.
        assert len(self._inputs) == 2
        assert self._inputs[1].shape.isScalar()
        assert len(self._outputs) == 2
        if self._inputs[0].shape.isUnknown():
            fail_str = 'SwitchOp {} propagateShapes unknown input shape' \
                       .format(self._name)
            raise NotImplementedError(fail_str)
        else:
            if self._inputs[0].shape != self._outputs[0].shape:
                raise NotImplementedError('SwitchOp propagateShapes {}'
                                          .format(self._name))
            self._outputs[0].shape.mergeShape(self._inputs[0].shape)
            if self._inputs[0].shape != self._outputs[1].shape:
                raise NotImplementedError('SwitchOp propagateShapes {}'
                                          .format(self._name))
            self._outputs[1].shape.mergeShape(self._inputs[0].shape)

    def calcAlgFlops(self):
        # SwitchOps perform no calculations
        return 0
