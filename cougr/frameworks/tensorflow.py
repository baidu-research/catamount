import os
import tensorflow as tf

# Tools to import Tensorflow MetaGraphs into CouGr format

TF_OP_TYPES = ['SaveV2', 'RestoreV2']
TF_OPS_TO_IGNORE = ['SaveV2', 'RestoreV2']

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
        if tf_node.type in TF_OP_TYPES:
            if tf_node.type in TF_OPS_TO_IGNORE:
                # print('WARN: Skipping unnecessary op {} type {}'
                #       .format(tf_node, tf_node.type))
                continue
            # [_] TODO: Map type to CouGr internal op type
            # [_] TODO: Create CouGr internal op
            # [_] TODO: Tensor creation (output?)? tf_node.outputs, shape def
            # [_] TODO: Link inputs from other ops
        else:
            print('WARN: Unknown op type: {}'.format(tf_node.type))

    return tf_graph
