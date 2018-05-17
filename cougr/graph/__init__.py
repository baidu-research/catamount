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

    @property
    def opsByName(self):
        return self._ops_by_name

    def isValid(self):
        ''' Return whether the graph is fully specified.
        '''
        # [_] TODO (Joel): Fill in here
        return True
