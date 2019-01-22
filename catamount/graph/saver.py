import pickle
import sys
# Very large compute graphs require that Python recurses the pointer chain
# to depths upward of 10000 (Python default is only 1000)
sys.setrecursionlimit(50000)


def saveGraph(graph, path):
    if path.split('.')[-1] != 'ccg':
        print('WARN: Catamount graphs should save to .ccg filenames')
    # Hacky way to store graphs for now
    # TODO: Change this to a protobuf implementation
    with open(path, 'wb') as outfile:
        pickle.dump(graph, outfile)

def loadGraph(path):
    if path.split('.')[-1] != 'ccg':
        print('WARN: Catamount graphs should have .ccg filenames')
    # Hacky way to load pickled graphs for now
    with open(path, 'rb') as infile:
        graph = pickle.load(infile)
    return graph
