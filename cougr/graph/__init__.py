from cougr.ops.base_op import Op
from cougr.ops.subgraph_op import SubgraphOp
from cougr.ops.placeholder import PlaceholderOp
from cougr.ops.variable import VariableOp


class GraphContextManagerHelper:
    def __init__(self):
        global _cougr_default_graph
        self._graph_stack_top = _cougr_default_graph

    def __enter__(self):
        pass

    def __exit__(self, type_arg, value_arg, traceback_arg):
        assert self._graph_stack_top is not None
        global _cougr_default_graph
        _cougr_default_graph  = self._graph_stack_top


class Graph(SubgraphOp):
    def __init__(self):
        super(Graph, self).__init__('graph')

    def asDefault(self):
        global _cougr_default_graph
        ctx_mgr = GraphContextManagerHelper()
        _cougr_default_graph = self
        return ctx_mgr

    def isValid(self):
        ''' Return whether the graph is fully specified. Check whether all ops
        have output tensors, whether those tensors have valid shapes, and
        whether their input and output tensors have producers and consumers
        specified. Then, check that sources and sinks are set up correctly.
        '''
        # Check op tensor producers and consumers
        for id, op in self._ops_by_name.items():
            assert op.parent is not None
            for in_tensor in op.inputs:
                if not in_tensor.isValid():
                    print('WARN: tensor {} not valid for op {}'
                          .format(in_tensor.name, op.name))
                    return False
                if op.name not in in_tensor.consumers.keys():
                    print('WARN: tensor {} not consumed by op {}'
                          .format(in_tensor.name, op.name))
                    return False
            for out_tensor in op.outputs:
                if not out_tensor.isValid():
                    print('WARN: tensor {} not valid for op {}'
                          .format(out_tensor.name, op.name))
                    return False
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

    def propagateTensorShapeNames(self):
        ''' Propagate bound tensor shape names through the network to bind
        downstream shapes.
        '''
        # Topologically traverse from sources to sinks. This can be a
        # flattened topological traversal from all sources to all sinks
        for next_op in self.getTopologicalOpOrder():
            next_op.propagateShapes()

    def bindTensorShapeDimensions(self, bind_dict):
        for name in bind_dict.keys():
            assert name in self._ops_by_name.keys()
            op = self._ops_by_name[name]
            assert type(op) == PlaceholderOp or \
                   type(op) == VariableOp
            for dim_idx, dim_name_or_symbol in enumerate(bind_dict[name]):
                if dim_name_or_symbol is not None:
                    op.bindTensorShapeDimension(dim_idx, dim_name_or_symbol)
        self.propagateTensorShapeNames()


# The CouGr default graph is used throughout the API
_cougr_default_graph = Graph()

def get_default_graph():
    global _cougr_default_graph
    return _cougr_default_graph

