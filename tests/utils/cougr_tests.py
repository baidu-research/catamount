import cougr
from cougr.graph import Graph
from cougr.ops import *
from cougr.tensors import *

def main():
    ''' Manually constructs a CouGr graph for a simplified word-level LSTM
    as described in Jozefowicz et al., Exploring the Limits of Language
    Modeling (here: https://arxiv.org/pdf/1602.02410.pdf).

    In particular, implement the following functionality:
    1) input_seq = placeholders(<batch_size * seq_len, vocab_size>)
       embedding_weights = variable(<vocab_size, hidden_dim>)
       lstm_seq = one_hot_lookup(embedding_weights, input_seq)
    2) for layer_id in range(num_layers):
           recur_input = placeholder(<batch_size, hidden_dim>)
           lstm_layer_weights = variable(<2*hidden_dim, 4*hidden_dim>)
           lstm_seq = lstm_layer(lstm_layer_weights, lstm_seq, recur_input)
    3) projection_weights = variable(<hidden_dim, proj_dim>)
       proj_seq = linear(projection_weights, lstm_seq)
    4) output_weights = variable(<proj_dim, vocab_size>)
       outputs = linear(output_weights, proj_seq)
    '''

    # Sizes of everything
    batch_size = None
    seq_length = None
    vocab_size = None
    hidden_dim = None
    num_layers = 2
    projection_dim = None

    # General flow for creating CouGr graph:
    # A) Create graph
    # B) Create op
    # C) Create op output tensor, then addOutput to op
    # D) Tell graph to connect op's inputs to prior op output tensors

    # Model definition parts:
    # 0) Create graph
    graph = Graph()
    # 1) Embedding layer
    # 2) Recurrent layers
    # 3) Projection layer
    # 4) Output layer
    proj_ph = PlaceholderOp('proj_ph')
    proj_seq = Tensor('proj_seq', TensorShape([batch_size, projection_dim]))
    proj_ph.addOutput(proj_seq)
    graph.addOp(proj_ph)

    output_var = VariableOp('output_weights')
    output_weights = Tensor(output_var.name,
                            TensorShape([projection_dim, vocab_size]))
    output_var.addOutput(output_weights)
    graph.addOp(output_var)

    output_op = MatMulOp('output_projection')
    out_tensor = Tensor(output_op.name,
                        TensorShape([batch_size, vocab_size]))
    output_op.addOutput(out_tensor)
    graph.addOp(output_op)
    graph.addInputToOp(output_op, proj_seq)
    graph.addInputToOp(output_op, output_weights)

    print(graph.calcAlgFlops())


if __name__ == "__main__":
    main()


