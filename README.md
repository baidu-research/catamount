# Catamount

Catamount is a compute graph analysis tool to load, construct, and modify deep learning models and to symbolically analyze their compute requirements.


## Getting Started

Do you want to learn about the compute requirements of the deep learning models you're training or inferencing?

To get a quick overview of Catamount and see how it works, you can follow this quick start guide:


### Use your favorite virtual environment or set one up

To create a virtual environment for Catamount:

```
python -m venv catamount_venv
```

Source the virtual environment

```
source catamount_venv/bin/activate
```


### Install Catamount requirements:

```
pip install -r requirements.txt
```

WARNING: This may change currently installed packages (e.g., Tensorflow), which may cause backward compatibility issues. If you have previously-installed packages, we recommend installing these requirements manually (e.g., `pip install <package>` with no version requirements).


### Install Catamount:

```
python setup.py install
```

WARNING: This may change currently installed Tensorflow to a particular minimum version. If you have an older version installed already, you can try removing the version dependency in `setup.py`.


### Quick test!

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
   * Symbolic algorithmic memory footprint (`"Alg mem footprint:"`). The total size of all tensors used to calculate one step of the graph.
   * Symbolic algorithmic minimal memory footprint (`"Alg minimal footprint"`). The minimum number of memory bytes that must be allocated to execute a training step. More precisely, it is the minimum---over all correct topological compute graph traversals---of the maximum memory capacity required to accommodate all active tensors during any step of the traversal. Active tensors are those produced by an op in a previous traversal step, but not yet consumed by each of its downstream ops.
   * Symbolic algorithmic input-output footprint (`"Alg IO footprint:"`). The amount of data accessed for input to and output from a model. Training data is often stored on disks, read from the disk, and placed into the model’s input memory allocations. Algorithmic IO is proportional to the batch size, but stays fixed as model size and training step compute requirements grow.
6. __A cool breakdown of algorithmic Flops, bytes, memory footprint, op-by-op:__


## How can I use Catamount?

We used Catamount to perform an extensive analysis of compute requirements across many deep learning applications. To get a sense for Catamount's power, we recommend reading our paper, ["Beyond Human-Level Accuracy: Computational Challenges in Deep Learning"](https://github.com/baidu-research/catamount/blob/master/reference/ppopp_2019_paper/PPoPP_2019_Projecting_Deep_Learning_Hardware_Requirements_Final.pdf) ([ACM link](https://dl.acm.org/citation.cfm?id=3295710)), published in the Principles and Practice of Parallel Programming (PPoPP), 2019.
