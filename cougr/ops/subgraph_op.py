from .base_op import Op


class SubgraphOp(Op):
    ''' A SubgraphOp designates a subgraph that manages a collection of ops.
        Note: SubgraphOps can contain other SubgraphOps (nesting).
    '''
    def __init__(self, name, ops_list=[]):
        super(SubgraphOp, self).__init__(name)
        self._ops_by_name = {}
        # Maintain a list of the ops that are sources to the graph. In
        # particular, if an op has no inputs or any of its inputs are
        # produced by ops outside the graph, then it is a source op.
        self._sources = {}
        # Maintain a list of the ops that are sinks from the graph. In
        # particular, if none of the op's outputs are consumed by any op
        # (i.e., terminal node) or they are consumed by other ops outside
        # the graph, then it is a sink op.
        self._sinks = {}

        for op in ops_list:
            self.addOp(op)
        self.findAllSourcesSinks()

    def debugString(self):
        to_return = 'In op {} of type {}:'.format(self._name, type(self))
        for op_name in sorted(self._ops_by_name.keys()):
            to_return += '\n Subop: {}'.format(op_name)
        return to_return

    def isValid(self):
        ''' Return whether the graph is fully specified. Check whether all ops
        have output tensors, whether those tensors have valid shapes, and
        whether their input and output tensors have producers and consumers
        specified. Then, check that sources and sinks are set up correctly.
        '''
        # Check op tensor producers and consumers
        for id, op in self._ops_by_name.items():
            self.debugAssert(op.parent is not None)
            if not op.isValid():
                return False
        # Check sources: Two conditions make an op a source:
        # 1) An op has no inputs, OR
        # 2) Some input must be produced outside block
        for id, op in self._sources.items():
            if len(op.inputs) > 0:
                some_external_input = False
                for in_tensor in op.inputs:
                    if in_tensor.producer.name not in self._ops_by_name.keys():
                        some_external_input = True
                if not some_external_input:
                    print('WARN: All inputs to op {} inside block!'
                          .format(op.name))
                    return False
        # Check sinks: Two conditions make an op a sink:
        # 1) An output has no consumers
        # 2) An output has consumers outside the block
        for id, op in self._sinks.items():
            if len(op.outputs) > 0:
                some_external_output = False
                for out_tensor in op.outputs:
                    if len(out_tensor.consumers) > 0:
                        for consumer in out_tensor.consumers.keys():
                            if consumer not in self._ops_by_name.keys():
                                some_external_output = True
                    else:
                        some_external_output = True
                if not some_external_output:
                    print('WARN: All outputs from op {} inside block!'
                          .format(op.name))
                    return False
        return True

    def addOp(self, op):
        self.debugAssert(isinstance(op, Op))
        self.debugAssert(op.name not in self._ops_by_name.keys())

        # Add the op
        self._ops_by_name[op.name] = op
        op.setParent(self)

        # Detect whether it is a true source or sink
        if len(op.inputs) == 0:
            self._sources[op.name] = op
        is_sink = True
        for out_tensor in op.outputs:
            if out_tensor.hasConsumers():
                is_sink = False
        if is_sink:
            self._sinks[op.name] = op

    def addInputToOp(self, op, tensor):
        self.debugAssert(op.name in self._ops_by_name.keys(),
                         'Op not in graph: {}'.format(op.name))
        op.addInput(tensor)
        tensor.addConsumer(op)
        if op.name in self._sources.keys():
            self.debugAssert(self._sources[op.name] == op)
            self._sources.pop(op.name)
        producer_op = tensor.producer
        if producer_op.name in self._sinks.keys():
            self.debugAssert(self._sinks[producer_op.name] == producer_op)
            self._sinks.pop(producer_op.name)

    def removeOp(self, op):
        # Remove op from _ops_by_name
        self._ops_by_name.pop(op.name, None)
        # Update sources as appropriate
        self._sources.pop(op.name, None)
        # Update sinks as appropriate
        self._sinks.pop(op.name, None)
        # Let the op disconnect itself from inputs
        op.resetInputs()

    @property
    def opsByName(self):
        return self._ops_by_name

    @property
    def inputs(self):
        # Collect the inputs to all sources and return
        to_return = set()
        for source_op in self._sources.values():
            for in_tensor in source_op.inputs:
                to_return.add(in_tensor)
        return list(to_return)

    @property
    def outputs(self):
        # Collect the outputs of all sinks and return
        to_return = set()
        for sink_op in self._sinks.values():
            for out_tensor in sink_op.outputs:
                for consumer in out_tensor.consumers.values():
                    to_return.add(out_tensor)
        return list(to_return)

    def outputShapeIllDefined(self):
        # Subgraph ops are collections of other ops. Ignore whether subgraph
        # ops have ill-defined output shapes in favor of just checking their
        # children ops directly.
        return False

    def findAllSourcesSinks(self):
        for op in self._ops_by_name.values():
            # Check if op is a source to the subgraph
            if op.name not in self._sources.keys():
                is_source = False
                for in_tensor in op.inputs:
                    if in_tensor.producer.name not in self._ops_by_name.keys():
                        is_source = True
                        break
                if is_source:
                    self._sources[op.name] = op
            # Check if the op is a sink of the subgraph
            if op.name not in self._sinks.keys():
                is_sink = False
                for out_tensor in op.outputs:
                    for consumer in out_tensor.consumers.keys():
                        if consumer not in self._ops_by_name.keys():
                            is_sink = True
                            break
                if is_sink:
                    self._sinks[op.name] = op

    def propagateShapes(self):
        # Propagating shapes is a flattened operation, so subgraphs
        # do not need to do any work for them
        pass

    # [_] TODO (Joel): Only traverse feeds to fetches and count along path
    def getTopologicalOpOrder(self, feed_dict=None, fetches_dict=None,
                              hierarchical=False):
        if feed_dict is not None:
            raise NotImplementedError(
                'Implement getTopologicalOpOrder to take feeds')

        if fetches_dict is not None:
            raise NotImplementedError(
                'Implement getTopologicalOpOrder to take fetches')

        topo_ordered_ops = []
        for source_op in self._sources.values():
            self.debugAssert(source_op.parent == self)
            topo_ordered_ops.append(source_op)
        visited_ops = set(topo_ordered_ops)
        frontier_ops = set()
        for op in topo_ordered_ops:
            for out_tensor in op.outputs:
                for op in out_tensor.consumers.values():
                    if op not in visited_ops:
                        frontier_ops.add(op)
        while len(frontier_ops) > 0:
            next_op = frontier_ops.pop()
            self.debugAssert(next_op.name in self._ops_by_name.keys(),
                             'Subgraph {}:\nOp not found! {}'.format(
                             self.name, next_op.debugString()))
            self.debugAssert(next_op not in visited_ops,
                             'Already visited {}!'.format(next_op.name))
            # Check if input producers have been visited
            if next_op.canVisit(visited_ops):
                visited_ops.add(next_op)
                for out_tensor in next_op.outputs:
                    for op in out_tensor.consumers.values():
                        if op.name in self._ops_by_name.keys():
                            if op not in visited_ops:
                                frontier_ops.add(op)
                            # Also get any first-level subgraphs
                            if op.parent != self and op.parent.parent == self:
                                if op.parent not in visited_ops:
                                    frontier_ops.add(op.parent)
                if not hierarchical or next_op.parent == self:
                    topo_ordered_ops.append(next_op)
            else:
                # Put the op back on the end of the frontier to check later
                frontier_ops.add(next_op)
        return topo_ordered_ops

    def calcModelParameters(self):
        ''' Calculate the number of model parameters for the subgraph.
        '''
        # Use a flattened traversal, since we only care about VariableOps
        ops_to_execute = self.getTopologicalOpOrder()
        total_model_params = 0
        for op in ops_to_execute:
            op_model_params = op.calcModelParameters()
            # print('Op: {}, alg_flops: {}'.format(op.name, op_alg_flops))
            total_model_params += op_model_params
        return total_model_params

    # [_] TODO (Joel): Only traverse feeds to fetches and count along path
    def calcAlgFlops(self, feed_dict=None, fetches_dict=None,
                     verbose=False):
        ''' Calculate the algorithmic Flops for the compute graph based on
        the ops that depend on ops in the feed_dict.
        '''
        # Use a hierarchical traversal and allow parents to count for their
        # children.
        ops_to_execute = self.getTopologicalOpOrder(feed_dict=feed_dict,
                             fetches_dict=fetches_dict, hierarchical=True)
        total_alg_flops = 0
        for op in ops_to_execute:
            self.debugAssert(op.parent == self,
                             'Incorrect parent for op {}: {}'
                             .format(op.name, op.parent.name))
            op_alg_flops = op.calcAlgFlops()
            if verbose:
                print('alg_flops {}: {}'.format(op.name, op_alg_flops))
            total_alg_flops += op_alg_flops
        return total_alg_flops

    # [_] TODO (Joel): Only traverse feeds to fetches and count along path
    def calcAlgBytes(self, feed_dict=None, fetches_dict=None,
                     verbose=False):
        ''' Calculate the algorithmic memory bytes accessed for the compute
        graph based on the ops that depend on ops in the feed_dict.
        '''
        # Use a hierarchical traversal and allow parents to count for their
        # children.
        ops_to_execute = self.getTopologicalOpOrder(feed_dict=feed_dict,
                             fetches_dict=fetches_dict, hierarchical=True)
        total_alg_bytes = 0
        for op in ops_to_execute:
            self.debugAssert(op.parent == self,
                             'Incorrect parent for op {}: {}'
                             .format(op.name, op.parent.name))
            op_alg_bytes = op.calcAlgBytes()
            if verbose:
                print('alg_bytes {}: {}'.format(op.name, op_alg_bytes))
            total_alg_bytes += op_alg_bytes
        return total_alg_bytes

    # [_] TODO (Joel): Only traverse feeds to fetches and count along path
    def calcAlgFootprint(self, feed_dict=None, fetches_dict=None,
                         verbose=False):
        ''' Calculate the algorithmic memory footprint accessed during a
        traversal of the compute graph.
        '''
        # Use a hierarchical traversal and allow parents to count for their
        # children.
        ops_to_execute = self.getTopologicalOpOrder(feed_dict=feed_dict,
                             fetches_dict=fetches_dict, hierarchical=True)
        total_alg_foot = 0
        for op in ops_to_execute:
            self.debugAssert(op.parent == self,
                             'Incorrect parent for op {}: {}'
                             .format(op.name, op.parent.name))
            op_alg_foot = op.calcAlgFootprint()
            if verbose:
                print('alg_foot {}: {}'.format(op.name, op_alg_foot))
            total_alg_foot += op_alg_foot
        return total_alg_foot

