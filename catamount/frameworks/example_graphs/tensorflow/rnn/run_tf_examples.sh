#!/bin/bash

echo "This script runs all the Tensorflow examples in this directory to create their graphs in the subdirectories here"
echo "Running..."


python tf_static_unroll.py
python tf_simple_while.py
python tf_dynamic_rnn.py
python tf_dynamic_rnn_with_backprop.py
