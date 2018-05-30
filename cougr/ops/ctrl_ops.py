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
        assert self._inputs[0].shape == self._outputs[0].shape

    def calcAlgFlops(self):
        # EnterOps perform no calculations
        return 0


class ExitOp(Op):
    ''' ExitOp designates the end of a control flow operation that acts
        on the input tensor to the op. ExitOps have no output tensors.
    '''
    def __init__(self, name):
        super(ExitOp, self).__init__(name)

    def propagateShapes(self):
        # ExitOps have no outputs to propagate to
        assert len(self._inputs) == 1
        assert len(self._outputs) == 0

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

    def propagateShapes(self):
        # MergeOps forward their input to their output for the
        # next iteration of a loop
        assert len(self._inputs) >= 1
        assert len(self._outputs) == 2

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
        assert len(self._outputs) == 2
        raise NotImplentedError('Must implement SwitchOp propagateShapes')

    def calcAlgFlops(self):
        # SwitchOps perform no calculations
        return 0
