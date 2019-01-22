import pickle


def saveGraph(graph, path):
    if '.ccg$' not in path:
        print('WARN: Catamount graphs should save to .ccg filenames')
    # Hacky way to store graphs for now
    with open(path, 'wb') as outfile:
        pickle.dump(graph, outfile)

def loadGraph(path):
    if '.ccg$' not in path:
        print('WARN: Catamount graphs should have .ccg filenames')
    # Hacky way to load pickled graphs for now
    with open(path, 'rb') as infile:
        graph = pickle.load(infile)
    return graph
