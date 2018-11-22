import sympy

import catamount
from catamount.tensors.tensor_shape import Dimension
from catamount.api import utils


# A set of helper functions for easy Flop calculations in tests
# Build symbol table for verifying Catamount Flops calculations
symbol_table = {}
subs_table = {}
correct_alg_flops = 0

def add_symbols(name, out_shape):
    # print('  Adding symbols for {}: out_shape: {}'.format(name, out_shape))
    global symbol_table
    global subs_table

    def add_symbol(symbol, dim):
        assert sym_name not in symbol_table.keys()
        symbol_table[sym_name] = symbol
        # print('Added symbol name {} with sym {}'.format(sym_name, symbol))
        if isinstance(dim, Dimension):
            dim = dim.value
        if dim is not None:
            subs_table[symbol] = dim

    if isinstance(out_shape, list):
        for idx, dim in enumerate(out_shape):
            sym_name = '{}::dim_{}'.format(name, idx)
            add_symbol(utils.getIntSymbolFromString(sym_name), dim)
    else:
        sym_name = '{}::unk'.format(name)
        add_symbol(utils.getIntSymbolFromString(sym_name), out_shape)

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


# Wrappers on Catamount ops to add their symbols to the test harness symbol table
def concat(name, out_shape, input_list, axis=0):
    add_symbols(name, out_shape)
    return catamount.concat(name, out_shape, input_list, axis)

def constant(name, out_shape, value=None):
    add_symbols(name, out_shape)
    return catamount.constant(name, out_shape, value)

def dynamic_stitch(name, out_shape, indices_list, data_list):
    add_symbols(name, out_shape)
    return catamount.dynamic_stitch(name, out_shape, indices_list, data_list)

def expanddims(name, out_shape, input, axis=0):
    add_symbols(name, out_shape)
    return catamount.expanddims(name, out_shape, input, axis)

def placeholder(name, out_shape):
    add_symbols(name, out_shape)
    return catamount.placeholder(name, out_shape)

def matmul(name, out_shape, in_a, in_b):
    add_symbols(name, out_shape)
    global correct_alg_flops
    in_a_dim_1 = '{}::dim_1'.format(in_a.name)
    out_dim_0 = '{}::dim_0'.format(name)
    out_dim_1 = '{}::dim_1'.format(name)
    correct_alg_flops += 2 * symbol_table[in_a_dim_1] * \
        symbol_table[out_dim_0] * symbol_table[out_dim_1]
    return catamount.matmul(name, out_shape, in_a, in_b)

def pointwise(name, op_type, out_shape, in_a, in_b=None):
    add_symbols(name, out_shape)
    global correct_alg_flops
    out_dim_0 = '{}::dim_0'.format(name)
    out_dim_1 = '{}::dim_1'.format(name)
    flops_per_elt = 1
    # Sigmoid and Tanh implementations perform specific number of Flops per
    # element as specified here.
    if op_type == catamount.SigmoidOp:
        flops_per_elt = 4
    if op_type == catamount.TanhOp:
        flops_per_elt = 6
    correct_alg_flops += flops_per_elt * symbol_table[out_dim_0] * \
                         symbol_table[out_dim_1]
    return catamount.pointwise(name, op_type, out_shape, in_a, in_b)

def reduce(name, op_func, out_shape, input, axes=0):
    add_symbols(name, out_shape)
    global correct_alg_flops
    to_add_flops = 1
    for idx in range(input.shape.rank):
        in_dim = '{}::dim_{}'.format(input.name, idx)
        to_add_flops *= symbol_table[in_dim]
    correct_alg_flops += to_add_flops
    return catamount.reduce(name, op_func, out_shape, input, axes)

def split(name, out_shape, input, size_splits=None, axis=0, num_split=2):
    if size_splits is not None:
        raise NotImplementedError('Helper split: Need to handle size_splits')
    for i in range(num_split):
        out_name = '{}_out{}'.format(name, i)
        add_symbols(out_name, out_shape)
    return catamount.split(name, out_shape, input, size_splits, axis, num_split)

def variable(name, out_shape):
    add_symbols(name, out_shape)
    return catamount.variable(name, out_shape)
