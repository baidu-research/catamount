import sympy

import cougr
from cougr.tensors.tensor_shape import Dimension


# A set of helper functions for easy Flop calculations in tests
# Build symbol table for verifying CouGr Flops calculations
symbol_table = {}
subs_table = {}
correct_alg_flops = 0

def add_symbols(name, out_shape):
    # print('  Adding symbols for {}: out_shape: {}'.format(name, out_shape))
    global symbol_table
    global subs_table
    for idx, dim in enumerate(out_shape):
        sym_name = '{}::dim_{}'.format(name, idx)
        symbol = sympy.Symbol(sym_name)
        assert sym_name not in symbol_table.keys()
        symbol_table[sym_name] = symbol
        # print('Added symbol name {} with sym {}'.format(sym_name, symbol))
        if isinstance(dim, Dimension):
            dim = dim.value
        if dim is not None:
            subs_table[symbol] = dim

def reset_symbols():
    global symbol_table
    global subs_table
    global correct_alg_flops
    symbol_table = {}
    subs_table = {}
    correct_alg_flops = 0

def get_subs_table():
    global subs_table
    return subs_table

def get_correct_alg_flops():
    global correct_alg_flops
    return correct_alg_flops


# Wrappers on CouGr ops to add their symbols to the test harness symbol table
def placeholder(name, out_shape):
    add_symbols(name, out_shape)
    return cougr.placeholder(name, out_shape)

def variable(name, out_shape):
    add_symbols(name, out_shape)
    return cougr.variable(name, out_shape)

def matmul(name, out_shape, in_a, in_b):
    add_symbols(name, out_shape)
    global correct_alg_flops
    in_a_dim_1 = '{}::dim_1'.format(in_a.name)
    out_dim_0 = '{}::dim_0'.format(name)
    out_dim_1 = '{}::dim_1'.format(name)
    correct_alg_flops += 2 * symbol_table[in_a_dim_1] * \
        symbol_table[out_dim_0] * symbol_table[out_dim_1]
    return cougr.matmul(name, out_shape, in_a, in_b)

def pointwise(name, op_type, out_shape, in_a, in_b=None):
    add_symbols(name, out_shape)
    global correct_alg_flops
    out_dim_0 = '{}::dim_0'.format(name)
    out_dim_1 = '{}::dim_1'.format(name)
    flops_per_elt = 1
    if op_type == cougr.SigmoidOp:
        flops_per_elt = 4
    if op_type == cougr.TanhOp:
        flops_per_elt = 6
    correct_alg_flops += flops_per_elt * symbol_table[out_dim_0] * \
                         symbol_table[out_dim_1]
    return cougr.pointwise(name, op_type, out_shape, in_a, in_b)

def reduce(name, op_func, out_shape, input, axes=0):
    add_symbols(name, out_shape)
    global correct_alg_flops
    to_add_flops = 1
    for idx in range(input.shape.rank):
        in_dim = '{}::dim_{}'.format(input.name, idx)
        to_add_flops *= symbol_table[in_dim]
    correct_alg_flops += to_add_flops
    return cougr.reduce(name, op_func, out_shape, input, axes)

def split(name, out_shape, input, num_splits=2, axis=0):
    for i in range(num_splits):
        out_name = '{}_out{}'.format(name, i)
        add_symbols(out_name, out_shape)
    return cougr.split(name, out_shape, input, num_splits, axis)

def concat(name, out_shape, input_list, axis=0):
    add_symbols(name, out_shape)
    return cougr.concat(name, out_shape, input_list, axis)

def constant(name, out_shape, axes=None):
    add_symbols(name, out_shape)
    return cougr.constant(name, out_shape, axes)
