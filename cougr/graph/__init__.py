from cougr.ops.base_op import Op
from cougr.ops.placeholder import PlaceholderOp
from cougr.ops.variable import VariableOp


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
        have output tensors, whether those tensors have valid shapes, and
        whether their input and output tensors have producers and consumers
        specified. Then, check that sources and sinks are set up correctly.
        '''
        # Check op tensor producers and consumers
        for id, op in self._ops_by_id.items():
            for in_tensor in op.inputs:
                if op.name not in in_tensor.consumers.keys():
                    print('WARN: tensor {} not consumed by op {}'
                          .format(in_tensor.name, op.name))
                    return False
            for out_tensor in op.outputs:
                if not out_tensor.isValid():
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
        # Topologically traverse from sources to sinks
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

    def getTopologicalOpOrder(self):
        topo_ordered_ops = list(self._sources.values())
        visited_ops = set(topo_ordered_ops)
        frontier_ops = []
        for op in topo_ordered_ops:
            for out_tensor in op.outputs:
                frontier_ops.extend(out_tensor.consumers.values())
        while len(frontier_ops) > 0:
            next_op = frontier_ops.pop(0)
            if next_op in visited_ops:
                # Skip ops that have been visited... make this smarter?
                continue
            # Check if input producers have been visited
            can_visit = True
            for in_tensor in next_op.inputs:
                if in_tensor.producer not in visited_ops:
                    can_visit = False
                    break
            if can_visit:
                visited_ops.add(next_op)
                for out_tensor in next_op.outputs:
                    frontier_ops.extend(out_tensor.consumers.values())
                topo_ordered_ops.append(next_op)
            else:
                # Put the op back on the end of the frontier to check later
                frontier_ops.append(next_op)
        return topo_ordered_ops

    # [_] TODO (Joel): Add fetches_dict. Only traverse feeds to fetches
    def getOpsToExecute(self, feed_dict=None, fetches_dict=None):
        if feed_dict is None and fetches_dict is None:
            # Must execute all ops
            return self._ops_by_name

        if fetches_dict is not None:
            raise NotImplementedError(
                'getOpsToExecute does not yet take fetches')

        ops_to_execute = {}
        # [_] TODO (Joel): Can we abstract this traversal to reuse it
        # in other parts of the code that want traversals?
        for feed_name, feed_op in feed_dict.items():
            assert type(feed_op) == PlaceholderOp
            ops_to_execute[feed_name] = feed_op
            # Traverse graph to find all downstream ops from feed_op
            # [_] TODO: NOTE: This fails for recurrent connections!
            frontier_ops = []
            for out_tensor in feed_op.outputs:
                frontier_ops.extend(out_tensor.consumers.values())
            while len(frontier_ops) > 0:
                next_op = frontier_ops.pop(0)
                if next_op is None:
                    continue
                ops_to_execute[next_op.name] = next_op
                for out_tensor in next_op.outputs:
                    frontier_ops.extend(out_tensor.consumers.values())
        return ops_to_execute

    # [_] TODO (Joel): Add fetches_dict. Only traverse feeds to fetches
    def calcAlgFlops(self, feed_dict=None, fetches_dict=None):
        ''' Calculate the algorithmic Flops for the compute graph based on
        the ops that depend on ops in the feed_dict.
        '''
        ops_to_execute = self.getOpsToExecute(feed_dict, fetches_dict)
        total_alg_flops = 0
        for op in ops_to_execute.values():
            op_alg_flops = op.calcAlgFlops()
            # print('Op: {}, alg_flops: {}'.format(op.name, op_alg_flops))
            total_alg_flops += op_alg_flops
        return total_alg_flops


# The CouGr default graph is used throughout the API
# [_] TODO (Joel): Make this managed! User should be able to
# set the default graph (without losing access to the default)
cougr_default_graph = Graph()

def get_default_graph():
    global cougr_default_graph
    return cougr_default_graph

