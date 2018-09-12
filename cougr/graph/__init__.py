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

    def __str__(self):
        # Dump the full graph definition
        # Note: This can be performed as a flattened operation
        out_str = ''
        for op_name in sorted(self._ops_by_name.keys()):
            op = self._ops_by_name[op_name]
            out_str += '{} {}\n'.format(op.name, op)
            for in_tensor in op._inputs:
                out_str += '  In tensor: {}\n'.format(in_tensor)
            for out_tensor in op._outputs:
                out_str += '  Out tensor: {}\n'.format(out_tensor)
        return out_str

    def asDefault(self):
        global _cougr_default_graph
        ctx_mgr = GraphContextManagerHelper()
        _cougr_default_graph = self
        return ctx_mgr

    def getPlaceholders(self):
        to_return = []
        for op in self._ops_by_name.values():
            if isinstance(op, PlaceholderOp):
                to_return.append(op)
        return to_return

    def propagateTensorShapeNames(self, warn_if_ill_defined=False,
                                  verbose=False):

        ''' Propagate bound tensor shape names through the network to bind
        downstream shapes.
        '''
        # Topologically traverse from sources to sinks. This can be a
        # flattened topological traversal from all sources to all sinks
        value_to_symbol_table = {}
        symbol_to_value_table = {}
        for op in self.getTopologicalOpOrder():
            if verbose:
                print('Before prop: {}'.format(op.debugString()))
            op.propagateShapes()
            if verbose:
                print('After prop: {}'.format(op.debugString()))

            if warn_if_ill_defined:
                # Check all op outputs to see if there are ill-defined
                # output shapes and warn if so:
                if op.outputShapeIllDefined():
                    print('WARN: Op out shape ill-defined: {} {}'
                          .format(op.name, op))
            for out_tensor in op.outputs:
                if out_tensor.shape.dims is None:
                    continue
                for dim in out_tensor.shape.dims:
                    if dim._value is not None and dim._symbol is not None:
                        if dim._value not in value_to_symbol_table:
                            value_to_symbol_table[dim._value] = set()
                        value_to_symbol_table[dim._value].add(dim._symbol)
                        if dim._symbol not in symbol_to_value_table:
                            symbol_to_value_table[dim._symbol] = set()
                        symbol_to_value_table[dim._symbol].add(dim._value)
        if verbose:
            print('Propagate Tensor Shape Symbols Complete')
            print('  Value to symbol table: {}'.format(value_to_symbol_table))
            print('  Symbol to value table: {}'.format(symbol_to_value_table))


    def bindTensorShapeDimensions(self, bind_dict, warn_if_ill_defined=False):
        for name in bind_dict.keys():
            assert name in self._ops_by_name.keys()
            op = self._ops_by_name[name]
            assert type(op) == PlaceholderOp or \
                   type(op) == VariableOp
            for dim_idx, dim_name_or_symbol in enumerate(bind_dict[name]):
                if dim_name_or_symbol is not None:
                    op.bindTensorShapeDimension(dim_idx, dim_name_or_symbol)
        self.propagateTensorShapeNames(warn_if_ill_defined)


# The CouGr default graph is used throughout the API
_cougr_default_graph = Graph()

def get_default_graph():
    global _cougr_default_graph
    return _cougr_default_graph

