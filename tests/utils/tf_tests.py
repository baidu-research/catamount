
import cougr.frameworks.tensorflow


tf_example_filename = 'cougr/frameworks/example_graphs/tensorflow/tf_example_graph.meta'

def run_tf_import_test():
    graph = cougr.frameworks.tensorflow.import_graph(tf_example_filename)


if __name__ == "__main__":
    run_tf_import_test()

