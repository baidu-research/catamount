import catamount.frameworks.tensorflow

from catamount.tests.utils.helpers import *
from catamount.ops.subgraph_op import SubgraphOp


graph_meta = 'catamount/frameworks/example_graphs/tensorflow/rnn/output_dynamic_rnn_with_backprop/tf_graph.meta'


def test_graph_traversal():
    # Import a graph to test out traversals
    graph = catamount.frameworks.tensorflow.import_graph(graph_meta)

    assert graph.isValid()

    # Get the topological op order for a flattened graph
    print('Testing flattened graph traversal')
    topo_ordered_ops = graph.getTopologicalOpOrder(hierarchical=False)
    # Check flattened-graph topological ordering
    visited_ops = set()
    for op in topo_ordered_ops:
        assert op.canVisit(visited_ops), \
               'Unable to visit op {}, visited_ops: {}' \
               .format(op.name, [v_op.name for v_op in visited_ops])
        visited_ops.add(op)

    # Get the topological op order for a hierarchical graph
    print('\n\nTesting hierarchical graph traversal')
    topo_ordered_ops = graph.getTopologicalOpOrder(hierarchical=True)
    # Check flattened-graph topological ordering
    visited_ops = set()
    for op in topo_ordered_ops:
        # All traversed ops in a hierarchical traversal must be the first
        # level ops under the specified graph. Thus, op.parent == graph
        assert op.parent == graph
        if not op.canVisit(visited_ops):
            # Traverse backward to find ancestors not visited
            bwd_frontier = set()
            for in_tensor in op.inputs:
                if in_tensor.producer not in visited_ops:
                    bwd_frontier.add(in_tensor.producer)
            unvisited_priors = set()
            while len(bwd_frontier) > 0:
                prior_op = bwd_frontier.pop()
                unvisited_priors.add(prior_op)
                for in_tensor in prior_op.inputs:
                    if in_tensor.producer not in visited_ops and \
                       in_tensor.producer not in unvisited_priors:
                        bwd_frontier.add(in_tensor.producer)
            assert op.canVisit(visited_ops), \
                'Illegal order traversing op {}\n  Missing priors: {}'.format(op.name, [prior_op.name for prior_op in unvisited_priors])
        visited_ops.add(op)
        if isinstance(op, SubgraphOp):
            for out_tensor in op.outputs:
                visited_ops.add(out_tensor.producer)


def test_graph_equal():
    graph_0 = catamount.frameworks.tensorflow.import_graph(graph_meta)
    graph_1 = catamount.frameworks.tensorflow.import_graph(graph_meta)
    # Reflexivity
    assert graph_0.isEqual(graph_0, verbose=True)
    # Symmetry
    assert graph_0.isEqual(graph_1, verbose=True)
    assert graph_1.isEqual(graph_0, verbose=True)

    # Modify graphs by hacking on a particular op
    my_op = graph_0.opsByName['a']
    my_op_1 = graph_0.opsByName['softmax/softmax']

    # Op naming
    old_name = my_op._name
    my_op._name = 'test_name_change'
    del graph_0._ops_by_name[old_name]
    graph_0._ops_by_name[my_op._name] = my_op
    assert (not graph_0.isEqual(graph_1, verbose=True))
    del graph_0._ops_by_name[my_op._name]
    my_op._name = old_name
    graph_0._ops_by_name[old_name] = my_op
    assert graph_0.isEqual(graph_1, verbose=True)

    # Op type: swap two op names
    old_name = my_op._name
    my_op._name = my_op_1._name
    my_op_1._name = old_name
    del graph_0._ops_by_name[my_op.name]
    del graph_0._ops_by_name[my_op_1.name]
    graph_0._ops_by_name[my_op.name] = my_op
    graph_0._ops_by_name[my_op_1.name] = my_op_1
    assert (not graph_0.isEqual(graph_1, verbose=True))
    old_name = my_op._name
    my_op._name = my_op_1._name
    my_op_1._name = old_name
    del graph_0._ops_by_name[my_op.name]
    del graph_0._ops_by_name[my_op_1.name]
    graph_0._ops_by_name[my_op.name] = my_op
    graph_0._ops_by_name[my_op_1.name] = my_op_1
    assert graph_0.isEqual(graph_1, verbose=True)

    # Finally, op removal
    graph_0.removeOp(my_op)
    assert (not graph_1.isEqual(graph_0, verbose=True))
    assert (not graph_0.isEqual(graph_1, verbose=True))


if __name__ == "__main__":
    test_graph_traversal()
    test_graph_equal()

