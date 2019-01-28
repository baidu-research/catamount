import sympy
from .base_op import Op
from ..api import utils

# HACK: Remove me later
from .stack_ops import StackPushOp


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
        # The ContextFrame that is associated with this subgraph. The
        # ContextFrame tracks the ops that gate the flow of tensors into
        # the subgraph.
        self._context_frame = None

        for op in ops_list:
            self.addOp(op)
        self.findAllSourcesSinks()

    def debugString(self):
        to_return = 'Op(name: {}, type: {}):'.format(self._name, type(self))
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

    def isEqual(self, other, verbose=False):
        if len(self.opsByName) != len(other.opsByName):
            if verbose:
                print('Graph equality: Different op count: self: {} other: {}'
                      .format(len(self.opsByName), len(other.opsByName)))
            return False
        for my_op in self.opsByName.values():
            if my_op.name not in other.opsByName.keys():
                if verbose:
                    print('Graph equality: Op not found in other!: {}'
                          .format(my_op.debugString()))
                return False
            if isinstance(my_op, SubgraphOp):
                continue
            other_op = other.opsByName[my_op.name]
            # Check op type
            if type(my_op) != type(other_op):
                if verbose:
                    print('Graph equality: Op not same type: {}\n{}'
                          .format(my_op.debugString(),
                                  other_op.debugString()))
                return False
            # Check inputs
            if len(my_op.inputs) != len(other_op.inputs):
                if verbose:
                    print('Graph equality: Inputs do not match: {}\n{}'
                          .format(my_op.debugString(),
                                  other_op.debugString()))
                return False
            for idx, in_tensor in enumerate(my_op.inputs):
                if in_tensor.shape != other_op.inputs[idx].shape:
                    if verbose:
                        print('Graph equality: In shapes do not match: {}\n{}'
                              .format(my_op.debugString(),
                                      other_op.debugString()))
                    return False
            # Check outputs
            if len(my_op.outputs) != len(other_op.outputs):
                if verbose:
                    print('Graph equality: Outputs do not match: {}\n{}'
                          .format(my_op.debugString(),
                                  other_op.debugString()))
                return False
            for idx, out_tensor in enumerate(my_op.outputs):
                if out_tensor.shape != other_op.outputs[idx].shape:
                    if verbose:
                        print('Graph equality: Out shapes do not match: {}\n{}'
                              .format(my_op.debugString(),
                                      other_op.debugString()))
                    return False
                for cons_name, my_consumer in out_tensor.consumers.items():
                    other_consumer = other_op.outputs[idx].consumers[cons_name]
                    if type(my_consumer) != type(other_consumer):
                        if verbose:
                            print('Graph equality: Out types do not match: '\
                                  '{}\n{}'.format(my_consumer.debugString(),
                                  other_consumer.debugString()))
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
        # Collect all source op input tensors that are produced by ops
        # from ancestor subgraphs
        to_return = set()
        for source_op in self._sources.values():
            for in_tensor in source_op.inputs:
                # Inputs are only those tensors produced by ops in my parent
                if in_tensor.producer.parent == self.parent:
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

    def setContextFrame(self, context_frame):
        self.debugAssert(self._context_frame is None)
        self._context_frame = context_frame

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

    def getFreeSymbols(self):
        to_return = super(SubgraphOp, self).getFreeSymbols()
        loop_iter_name = '{}::iters'.format(self.name)
        loop_iters = utils.getIntSymbolFromString(loop_iter_name)
        to_return.add(loop_iters)
        return to_return

    def propagateShapes(self, make_symbolic=False):
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
        op_inputs_visited = {}
        frontier_ops = set()
        visited_ops = set()
        # Prime the frontier with source ops
        # TODO (Joel): Could set frontier equal to feed_dict?
        for source_op in self._sources.values():
            self.debugAssert(source_op.parent == self)
            self.debugAssert(source_op not in op_inputs_visited)
            op_inputs_visited[source_op] = set()
            for in_tensor in source_op.inputs:
                self.debugAssert(self.parent is not None)
                # If the producer op is not from this subgraph, it needs
                # to be visited in order to prime the subgraph
                if in_tensor.producer.parent != self:
                    op_inputs_visited[source_op].add(in_tensor.producer)
                    visited_ops.add(in_tensor.producer)
            if source_op.canVisit(op_inputs_visited[source_op]):
                frontier_ops.add(source_op)
        # Continually visit frontier ops until none left
        while len(frontier_ops) > 0:
            next_op = frontier_ops.pop()
            self.debugAssert(next_op.canVisit(visited_ops),
                             'Next op {} cannot visit. Visited: {}'
                             .format(next_op.name, visited_ops))
            if not hierarchical or next_op.parent == self:
                topo_ordered_ops.append(next_op)
            visited_ops.add(next_op)
            for out_tensor in next_op.outputs:
                for consumer in out_tensor.consumers.values():
                    if consumer in visited_ops:
                        continue
                    if consumer not in op_inputs_visited:
                        op_inputs_visited[consumer] = set()
                    # To handle subgraph ops, the producer of a tensor must
                    # be the op added to the consumer's scoreboard. Also, the
                    # producer needs to be added to visited_ops.
                    producer_op = next_op
                    if next_op != out_tensor.producer:
                        self.debugAssert(isinstance(next_op, SubgraphOp))
                        if hierarchical:
                            producer_op = out_tensor.producer
                            visited_ops.add(producer_op)
                    op_inputs_visited[consumer].add(producer_op)
                    # Check if the consumer can now be visited, and if so,
                    # add it to the frontier
                    if consumer.canVisit(op_inputs_visited[consumer]):
                        if not hierarchical or consumer.parent == self:
                            frontier_ops.add(consumer)
                    # Also check the consumer to see if its parent subgraph
                    # can be traversed (if parent different from self)
                    if consumer.parent != self and \
                       consumer.parent not in visited_ops:
                        if consumer.parent.canVisit(visited_ops):
                            frontier_ops.add(consumer.parent)
            # ----------------------------------------------------------------
            # HACK! StackPushOps need to signal that the corresponding
            # StackPopOp may now be ready to visit. If so, add it to the
            # frontier.
            # TODO: Replace this check with control dependencies?
            if isinstance(next_op, StackPushOp):
                stack_pop_op = next_op._stack._pop
                self.debugAssert(stack_pop_op not in visited_ops)
                if stack_pop_op not in op_inputs_visited:
                    op_inputs_visited[stack_pop_op] = set()
                # Signal that the stack has been visited by adding the
                # StackPushOp to the StackPopOp's visited inputs
                op_inputs_visited[stack_pop_op].add(next_op)
                if stack_pop_op.canVisit(op_inputs_visited[stack_pop_op]):
                    if not hierarchical or stack_pop_op.parent == self:
                        frontier_ops.add(stack_pop_op)
            # ----------------------------------------------------------------
        # print('Subgraph: {}'.format(self.name))
        # print('All ops: {}'.format(len(self._ops_by_name.keys())))
        children_ops = set()
        for op in self._ops_by_name.values():
            if op.parent == self:
                children_ops.add(op)
        # print('Children ops: {}'.format(len(children_ops)))
        topo_set = set(topo_ordered_ops)
        # print('Topo: {} (set: {})'.format(len(topo_ordered_ops), len(topo_set)))
        # print('Visited: {}'.format(len(visited_ops)))
        # print('Op ins visited: {}'.format(len(op_inputs_visited.keys())))
        for op in self._ops_by_name.values():
            if op not in visited_ops:
                if not hierarchical or op.parent == self:
                    print('  Not visited: {}'.format(op.name))
        # Some sanity checks after traversal
        # Subgraphs can have inputs (visited) from hierarchical parents
        subgraph_ops = set(self._ops_by_name.values())
        if hierarchical:
            self.debugAssert(visited_ops.issuperset(children_ops))
        else:
            self.debugAssert(visited_ops == subgraph_ops)
        self.debugAssert(visited_ops.issuperset(topo_ordered_ops))
        topo_minus_subgraph = topo_set.difference(subgraph_ops)
        self.debugAssert(subgraph_ops.issuperset(topo_ordered_ops),
                         'Ops in topo not in subgraph: {}'
                         .format([op.name for op in topo_minus_subgraph]))
        return topo_ordered_ops

    def calcModelParameters(self):
        ''' Calculate the number of model parameters for the subgraph.
        '''
        # Use an arbitrary flat traversal, since only care about VariableOps
        ops_to_execute = self._ops_by_name.values()
        total_model_params = 0
        for op in ops_to_execute:
            if isinstance(op, SubgraphOp):
                # Flat traversal, so do not recurse into subgraphs
                continue
            op_model_params = op.calcModelParameters()
            # print('Subgraph: {}, Op: {}, Params: {}'
            #       .format(self.name, op.name, op_model_params))
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

    # [_] TODO (Joel): Only traverse feeds to fetches and count along path
    def calcMinimalFootprint(self, feed_dict=None, fetches_dict=None,
                             verbose=False, symbol_subs=None):
        ''' Calculate the minimal memory footprint accessed during a
        traversal of the compute graph.
        '''
        max_footprint = 0
        curr_footprint = 0
        # A scoreboard to track the consumption of tensors during traversal
        tensors_to_consume = {}
        visited_ops = set()
        max_footprint, curr_footprint = self.calcMinimalFootprintSub(
                                            max_footprint, curr_footprint,
                                            tensors_to_consume, visited_ops,
                                            verbose=verbose,
                                            symbol_subs=symbol_subs)
        return max_footprint

    def calcMinimalFootprintSub(self, max_footprint, curr_footprint,
                                tensors_to_consume, visited_ops,
                                verbose=False, symbol_subs=None):
        # NOTE: This function is currently an approximation for subgraphs!
        # TODO (Joel): Figure out how to pass feeds and fetches?
        # TODO (Joel): Move this out to the loop control block op!
        # The maximum footprint size for a subgraph is approximately equal
        # to the maximum of the following footprints:
        #   1) The maximum footprint while executing any single iteration
        #   2) The maximum footprint before the start of the subgraph plus
        #      (Number of iterations) * (Change in footprint from the
        #      start to the end of a single iteration)
        #   3) For If-blocks, the maximum footprint of either the True or
        #      False paths, multipled by an indicator for whether the If
        #      statement evaluated to True or False, respectively.
        # print('Starting traversal: {}'.format(self.name))
        ops_to_execute = self.getTopologicalOpOrder(hierarchical=True)
        my_visited_ops = set()
        for op in self._sources.values():
            for in_tensor in op.inputs:
                if in_tensor.producer.parent != self:
                    assert in_tensor.producer not in self._ops_by_name.keys()
                    my_visited_ops.add(in_tensor.producer)
        my_max_footprint = max_footprint
        my_curr_footprint = curr_footprint
        for op in ops_to_execute:
            self.debugAssert(op.canVisit(my_visited_ops),
                             'Unable to visit op {}, visited_ops: {}'
                             .format(op.name,
                                     [v_op.name for v_op in my_visited_ops]))
            self.debugAssert(op.canVisit(visited_ops),
                             'Cannot visit {}!'.format(op.name))
            my_max_footprint, my_curr_footprint = op.calcMinimalFootprintSub(
                                                      my_max_footprint,
                                                      my_curr_footprint,
                                                      tensors_to_consume,
                                                      visited_ops,
                                                      verbose=verbose,
                                                      symbol_subs=symbol_subs)
            if op.calcAlgFootprint() != 0:
                # If the op receives some inputs from outside the subgraph,
                # restore those inputs into the footprint to ensure that they
                # will not incorrectly negatively impact the min footprint
                readd_input_sizes = 0
                for in_tensor in op.inputs:
                    if in_tensor.producer.parent != self:
                        readd_input_sizes += in_tensor.size
                if readd_input_sizes != 0:
                    my_curr_footprint += readd_input_sizes
                    my_max_footprint = utils.getSymbolicMaximum(
                                           my_curr_footprint,
                                           my_max_footprint,
                                           symbol_subs)
            my_visited_ops.add(op)
            if isinstance(op, SubgraphOp):
                for out_tensor in op.outputs:
                    my_visited_ops.add(out_tensor.producer)
        # TODO (Joel): THIS IS THE CALCULATION FOR A LOOP SUBGRAPH. MUST
        # MOVE TO LOOP CONDITION OP AND CHANGE THIS FUNCTION TO A
        # NOTIMPLEMENTED ERROR
        loop_iter_name = '{}::iters'.format(self.name)
        loop_iters = utils.getIntSymbolFromString(loop_iter_name)
        my_curr_footprint = curr_footprint + \
                            (my_curr_footprint - curr_footprint) * loop_iters
        my_max_footprint = utils.getSymbolicMaximum(my_max_footprint,
                                                    my_curr_footprint,
                                                    symbol_subs)
        if verbose:
            if isinstance(my_curr_footprint, sympy.Expr):
                my_int_curr_foot = my_curr_footprint.subs(symbol_subs)
            else:
                my_int_curr_foot = my_curr_footprint
            print('  FOOT: {} {} {}'.format(self.name, my_max_footprint,
                                            my_int_curr_foot))
        return my_max_footprint, my_curr_footprint

