import os
import tensorflow as tf

from cougr.ops.placeholder import PlaceholderOp

# Tools to import Tensorflow MetaGraphs into CouGr format

TF_TO_COUGR_MAP = {
    'SaveV2': None, # Ignore Saver ops
    'RestoreV2': None, # Ignore Restore ops
    'Placeholder': PlaceholderOp,
}

def import_graph(tf_filename):
    if '.meta' not in tf_filename or not os.path.exists(tf_filename):
        raise FileNotFoundError('ERROR: Invalid file {}. Must be .meta file'
            .format(tf_filename))
    saver = tf.train.import_meta_graph(tf_filename)
    sess = tf.Session()
    try:
        tf_model_name = tf_filename.replace('.meta', '')
        saver.restore(sess, tf_model_name)
    except Exception:
        print('WARN: Cannot find checkpoint data {}, trying to proceed'
              .format('{}.data-?????-of-?????'.format(tf_filename)))

    tf_graph = sess.graph
    graph = [] # For now, just a list of nodes
    for tf_node_name in tf_graph._nodes_by_name.keys():
        tf_node = tf_graph._nodes_by_name[tf_node_name]
        if tf_node.type in TF_TO_COUGR_MAP.keys():
            # Map to CouGr op type
            cougr_type = TF_TO_COUGR_MAP[tf_node.type]
            if cougr_type is None:
                # print('WARN: Skipping unnecessary op {} type {}'
                #       .format(tf_node, tf_node.type))
                continue
            # Create the CouGr internal op
            node = cougr_type(tf_node_name)
            # [_] TODO: Tensor creation (output?)? tf_node.outputs, shape def
            # [_] TODO: Link inputs from other ops? Or on second pass?
            graph.append(node)
        else:
            print('WARN: Unknown op type: {}'.format(tf_node.type))

    print(graph)

    return tf_graph, graph
