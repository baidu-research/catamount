# Catamount

Catamount is a compute graph analysis tool to load, construct, and modify deep learning models and to symbolically analyze their compute requirements.


## Getting Started

   Do you want to learn about the compute requirements of the deep learning models you're training or inferencing?

   To get a quick overview of Catamount and see how it works, you can follow this quick start guide:


   ### 1. Use your favorite virtual environment or set one up

   To create a virtual environment for Catamount:

   ```
   python -m venv catamount_venv
   ```

   Source the virtual environment

   ```
   source catamount_venv/bin/activate
   ```


   ### 2. Install Catamount requirements:

   ```
   pip install -r requirements.txt
   ```

   WARNING: This may change currently installed packages (e.g., Tensorflow), which may cause backward compatibility issues. If you have previously-installed packages, we recommend installing these requirements manually (e.g., `pip install <package>` with no version requirements).


   ### 3. Install Catamount:

   ```
   python setup.py install
   ```

   WARNING: This may change currently installed Tensorflow to a particular minimum version. If you have an older version installed already, you can try removing the version dependency in `setup.py`.


   ### 4. Quick example!

   A quick and interesting test to run is the basic LSTM language model. You can run it with the following command:

   ```
   python catamount/tests/full/tf_language_models.py --domain wordlm
   ```

   We recommend piping the output to a file and browsing the file afterward.

   NOTE: If you have tensorflow-gpu installed and use NVIDIA GPUs, you might want to prepend this command with `CUDA_VISIBLE_DEVICES=""` to make sure TF does not allocate your GPUs to this task---no GPUs are necessary.


   The output contains the following parts

   0. Warnings: just a note that Catamount errs on the side of verbose output to make sure that you are aware if it is running off the rails (we may add a verbosity option later). Warnings in tests can be safely ignored.
   1. An initial print-out of the loaded Tensorflow graph (`"Initial graph:"`). Each op is listed along with its input and output tensors.
   2. Initial analysis of the the graph:
      * `"Initial parameters"`: The number of model parameters given the particular numerical dimensions in the example
      * `"Placeholders"`: A list of the ops that represent inputs to the graph
   3. A note that the code is binding variable names to tensor dimensions and propagating them through the graph (`"Binding variables"`)
   4. The cleaned graph after dimension name propagation. Each op now shows the symbolic name for tensor dimensions.
   5. __The stuff you came for:__
      * Symbolic parameter count (`"Parameters:"`). The number of parameters (weights) in the model based on hyperparameters (e.g., hidden dimension, vocabulary size).
      * Symbolic algorithmic Flop count (`"Algorithmic Flops:"`). The number of FLOPs required to perform the mathematical calculation of a compute graph op (note: either floating point or integer arithmetic). For example, __algorithmic FLOPs__ include the multiplies and accumulations in a matrix multiply op. __Algorithmic FLOPs__ *do not* include other instructions executed by hardware to perform the computation, such as address, loop invariant, or branch target calculations. Hardware instructions that are not counted in algorithmic FLOPs are likely to account for at most constant overhead per algorithmic FLOP.
      * Symbolic algorithmic memory bytes accessed (`"Alg bytes accessed:"`). The total memory bytes that an op must read as inputs and write as outputs to perform the operation. __Algorithmic op bytes__ *do not* include intermediate data or other memory that might be used to perform the operations, and ignores hardware effects such as caching.
      * Symbolic algorithmic memory footprint (`"Alg mem footprint:"`). The total size of all tensors used to perform one iteration of the graph.
      * Symbolic algorithmic minimal memory footprint (`"Alg minimal footprint"`). The minimum number of memory bytes that must be allocated to execute a training step. More precisely, it is the minimum---over all correct topological compute graph traversals---of the maximum memory capacity required to accommodate all active tensors during any step of the traversal. Active tensors are those produced by an op in a previous traversal step, but not yet consumed by each of its downstream ops.
      * Symbolic algorithmic input-output footprint (`"Alg IO footprint:"`). The amount of data accessed for input to and output from a model. Training data is often stored on disks, read from the disk, and placed into the model’s input memory allocations. Algorithmic IO is proportional to the batch size, but stays fixed as model size and training step compute requirements grow.
   6. __A cool op-by-op breakdown of algorithmic Flops, bytes, memory footprint__


## How does Catamount work?

At a high level, Catamount is just a simple tool for constructing, modifying, and analyzing compute graphs. These compute graphs include nodes, or "ops", that perform a mathematical computation---e.g., matrix-vector multiplication, convolution, or pointwise operations---on input data. Data is passed between ops using "tensors" (like data arrays) that encode the data’s structure and dependencies between ops.

We recommend inspecting the example code (used in the Quick Start above) in [`catamount/tests/full/tf_language_models.py`](https://github.com/baidu-research/catamount/blob/master/catamount/tests/full/tf_language_models.py). Briefly, this code does the following:

First, it loads an LSTM word-level language model from a Tensorflow checkpoint. Then, it names all the dimensions for weights, inputs ("placeholders"), and constants, and then propagates these names through the graph. Afterward, it calculates algorithmic compute requirements. For example, you'll find a couple lines like the following (`graph.calcAlgFlops()` in code):
```
Algorithmic Flops: Model/Gradient/Compute/gradients/b_count_2_block::iters*(32*hidden_dim**2*subbatch_size + 8*hidden_dim**2 + 37*hidden_dim*subbatch_size + 4*hidden_dim + 3) + Model/Gradient/Compute/gradients/b_count_6_block::iters*(32*hidden_dim**2*subbatch_size + 8*hidden_dim**2 + 37*hidden_dim*subbatch_size + 4*hidden_dim + 3) + Model/Recurrent_1_lstm_1/rnn/while/LoopCond_block::iters*(16*hidden_dim**2*subbatch_size + 33*hidden_dim*subbatch_size + 6) + Model/Recurrent_2_lstm_1/rnn/while/LoopCond_block::iters*(16*hidden_dim**2*subbatch_size + 33*hidden_dim*subbatch_size + 6) + 48*hidden_dim**2 + 6*hidden_dim*sequence_length*subbatch_size*vocab_size + 4*hidden_dim*sequence_length*subbatch_size + 3*hidden_dim*vocab_size + 24*hidden_dim + 7*sequence_length*subbatch_size*vocab_size + 4*sequence_length*subbatch_size + 3*vocab_size + 53
With specified dims: 2597058084257
```

This output shows that there are 2 recurrent layers in the model's forward pass (Model/Recurrent_1_lstm_1/rnn/while, and Model/Recurrent_2_lstm_1/rnn/while), and their respective backward passes, each with different compute ops. Then there are a few other non-recurrent layers (embeddings, output layer GEMM and biases, loss, etc.) with different compute requirements. All of these formulas are symbolic (using the Python package `sympy`), and you can evaluate them with particular values by running something like `graph.calcAlgFlops().subs({'batch_size': 32, 'hidden_dim': 1024, ...})`

The tool has functionality for algorithmic FLOPs, memory bytes accessed in an iteration, the total memory footprint (total bytes of all tensors that must be used during an iteration), and an estimate of a minimal memory footprint assuming that tensors can be freed from memory after their last use.


## What models are currently included with Catamount?

Catamount includes a number of existing models (loaded from Tensorflow graphs) that you can analyze:

1. Character language modeling with a Recurrent Highway Network (RHN): `python catamount/tests/full/tf_language_models.py --domain charlm`
2. Word language modeling with an LSTM: `python catamount/tests/full/tf_language_models.py --domain wordlm`
3. The Google NMT hybrid attention recurrent model: `python catamount/tests/full/tf_language_models.py --domain nmt`
4. A few ResNets (of varying depth, filter scaling---proportional to the baseline number of filters): `python catamount/tests/full/tf_image_resnet.py --help`
5. The Word2Vec implementation in Tensorflow tutorials: `python catamount/tests/full/tf_word2vec.py`
6. A speech recognition model, recurrent encoder-decoder with attention: `python catamount/tests/full/tf_speech_attention.py`
7. The BERT word-level language model: `python catamount/tests/full/tf_bert_language_models.py --model_name cased_L-12_H-768_A-12`

More may be added later! If you set up a graph, please contribute it back!


## How can I use Catamount?

We used Catamount to perform an extensive analysis of compute requirements across many deep learning applications. To get a sense for Catamount's power, we recommend reading our paper, ["Beyond Human-Level Accuracy: Computational Challenges in Deep Learning"](https://github.com/baidu-research/catamount/blob/master/reference/ppopp_2019_paper/PPoPP_2019_Projecting_Deep_Learning_Hardware_Requirements_Final.pdf) ([ACM link](https://dl.acm.org/citation.cfm?id=3295710)), published in the Principles and Practice of Parallel Programming (PPoPP), 2019. See the paper artifact (pp. 13-14) for more details for running Catamount.


## What haven't we covered above... (a lot)

   1. Catamount has a (preliminary) programming API that allows you to manually specify graphs. This API is intended to look much like Tensorflow or PyTorch code. See `catamount/api`.
   2. To run all the regression tests, simply execute `bash catamount/tests/run_tests.sh`. All tests should pass if (when ;) you contribute code back to Catamount.
   3. Catamount chooses a "generously large" graph-level intermediate representation that maps many Tensorflow ops. We expect it will be easily adaptable to load from other frameworks (e.g., ONNX, PyTorch, or MXNet). We also expect it should be easy to add graph-level transforms for op-level fusion or fission (e.g., to transform from an LSTM cell op to constituent GEMM and pointwise operations).
   4. Given the computational requirements for each op in a Catamount graph, one can trivially estimate ["Roofline" level performance](https://people.eecs.berkeley.edu/~kubitron/courses/cs252-S12/handouts/papers/RooflineVyNoYellow.pdf) for each op and for the whole graph. Further performance estimates can be achieved by extending the graph definition with hardware-implementation-specific performance models.

