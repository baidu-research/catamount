from cougr.ops.base_op import Op

class Graph:
    def __init__(self):
        self._next_op_id = 0
        self._ops_by_name = {}
        self._ops_by_id = {}
        # Maintain a list of the ops that are sources to the graph
        # (in particular, ops with no inputs must be sources)
        self._sources = {}
        # Maintain a list of the ops that are sinks from the graph
        # (in particular, ops whose outputs have no consumers)
        self._sinks = {}

    def addOp(self, op):
        assert isinstance(op, Op)
        assert op.name not in self._ops_by_name.keys()
        self._ops_by_id[self._next_op_id] = op
        self._next_op_id += 1
        self._ops_by_name[op.name] = op
        if len(op.inputs) == 0:
            self._sources[op.name] = op
        is_sink = True
        for out_tensor in op.outputs:
            if out_tensor.hasConsumers():
                is_sink = False
        if is_sink:
            self._sinks[op.name] = op

    def getOpByName(self, op_name):
        return self._ops_by_name[op_name]

    def addInputToOp(self, op, tensor):
        op.addInput(tensor)
        tensor.addConsumer(op)
        if op.name in self._sources.keys():
            assert self._sources[op.name] == op
            self._sources.pop(op.name)
        producer_op = tensor.producer
        if producer_op.name in self._sinks.keys():
            assert self._sinks[producer_op.name] == producer_op
            self._sinks.pop(producer_op.name)

    @property
    def opsByName(self):
        return self._ops_by_name

    def isValid(self):
        ''' Return whether the graph is fully specified. Check whether all ops
        have output tensors and whether their input and output tensors have
        producers and consumers specified. Then, check that sources and sinks
        are set up correctly.
        '''
        # Check op tensor producers and consumers
        for id, op in self._ops_by_id.items():
            for in_tensor in op.inputs:
                if op.name not in in_tensor.consumers.keys():
                    print('WARN: tensor {} not consumed by op {}'
                          .format(in_tensor.name, op.name))
                    return False
            for out_tensor in op.outputs:
                if out_tensor.producer is not op:
                    print('WARN: tensor {} not produced by op {}'
                          .format(out_tensor.name, op.name))
                    return False
        # Check sources and sinks
        for id, op in self._sources.items():
            if len(op.inputs) > 0:
                print('WARN: op {} is not a true source'
                      .format(op.name))
                return False
        for id, op in self._sinks.items():
            for out_tensor in op.outputs:
                if len(out_tensor.consumers) > 0:
                    print('WARN: op {} is not a true sink: {}'
                          .format(op.name, out_tensor.name))
                    return False
        return True

    # [_] TODO (Joel): add code that allows marking the nodes to be executed
    # [_] TODO (Joel): add subroutine that only traverses marked nodes
    def calcAlgFlops(self):
        total_alg_flops = 0
        for id, op in self._ops_by_id.items():
            op_alg_flops = op.calcAlgFlops()
            # print('Op: {}, alg_flops: {}'.format(op.name, op_alg_flops))
            total_alg_flops += op_alg_flops
        return total_alg_flops
