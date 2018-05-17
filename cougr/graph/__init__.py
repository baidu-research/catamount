from cougr.ops.base_op import Op

class Graph:
    def __init__(self):
        self._next_op_id = 0
        self._ops_by_name = {}
        self._ops_by_id = {}

    def addOp(self, op):
        assert isinstance(op, Op)
        assert op.name not in self._ops_by_name.keys()
        self._ops_by_id[self._next_op_id] = op
        self._next_op_id += 1
        self._ops_by_name[op.name] = op

    def getOpByName(self, op_name):
        return self._ops_by_name[op_name]

    def addInputToOp(self, op, tensor):
        op.addInput(tensor)
        tensor.addConsumer(op)

    @property
    def opsByName(self):
        return self._ops_by_name

    def isValid(self):
        ''' Return whether the graph is fully specified. Check whether all ops
        have output tensors and whether their input and output tensors have
        producers and consumers specified.
        '''
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
        return True
