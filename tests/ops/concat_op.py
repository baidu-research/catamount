import sympy

import cougr
from cougr.graph import Graph
from cougr.tensors.tensor_shape import Dimension, TensorShape


# Build symbol table with this function (for verifying CouGr
# Flops calculations)
symbol_table = {}
subs_table = {}
correct_alg_flops = 0

def add_symbols(name, out_shape):
    global symbol_table
    global subs_table
    for idx, dim in enumerate(out_shape):
        sym_name = '{}::dim_{}'.format(name, idx)
        symbol = sympy.Symbol(sym_name)
        assert sym_name not in symbol_table.keys()
        symbol_table[sym_name] = symbol
        # print('Added symbol name {} with sym {}'.format(sym_name, symbol))
        if dim is not None:
            subs_table[symbol] = dim


def reset_symbols():
    global symbol_table
    global subs_table
    global correct_alg_flops
    symbol_table = {}
    subs_table = {}
    correct_alg_flops = 0


def placeholder(name, out_shape):
    add_symbols(name, out_shape)
    return cougr.placeholder(name, out_shape)


def concat(name, out_shape, input_list, axis=0):
    add_symbols(name, out_shape)
    return cougr.concat(name, out_shape, input_list, axis)


def test_concat_op():
    ''' Specify graphs with concat operations and make sure dimensions behave
    as desired. 
    '''

    combos = [([[None, None], [None, None]], 0),
              ([[None, None], [None, None]], 1),
              ([[3, None], [3, None]], 0),
              ([[3, None], [3, None]], 1),
              ([[3, 7], [6, 7]], 0),
              ([[3, 15], [3, None]], 1),
              ([[3, None, 7, 15], [3, 15, 7, None]], 0),
              ([[3, None, 7, 15], [3, 15, 7, None]], 1),
              ([[3, None, 7, 15], [3, 15, 7, None]], 2),
              ([[3, None, 7, 15], [3, 15, 7, None]], 3),
              ([[3, None, 7, 15], [3, 15, 7, None], [None, 15, 7, 30]], 3)]

    for combo in combos:
        graph = Graph()
        with graph.asDefault():
            ph_dims, axis = combo
            print('Testing concat with in dims {}, axis {}'
                  .format(ph_dims, axis))

            # Build model
            in_phs = []
            rank = None
            for idx, ph_dim in enumerate(ph_dims):
                ph_name = 'in_{}'.format(idx)
                in_phs.append(placeholder(ph_name, ph_dim))
                if rank is None:
                    rank = in_phs[idx].shape.rank
                else:
                    assert rank == in_phs[idx].shape.rank
            concat_out = concat('concat', [None] * rank, in_phs, axis=axis)

            assert graph.isValid()

            feed_dict = {}
            out_c_dim = Dimension(0)
            for in_ph, ph_dim in zip(in_phs, ph_dims):
                in_ph_dims = []
                for idx, dim in enumerate(ph_dim):
                    append_dim_sym = None
                    if dim is None:
                        dim_name = 'bind_{}_{}'.format(in_ph.name, idx)
                        append_dim_sym = sympy.Symbol(dim_name)
                    else:
                        append_dim_sym = dim
                    in_ph_dims.append(append_dim_sym)
                    if idx == axis:
                        append_dim = Dimension(None)
                        append_dim.setSymbolOrName(append_dim_sym)
                        out_c_dim += append_dim
                feed_dict[in_ph.name] = in_ph_dims
            print('    Feed dict: {}'.format(feed_dict))

            graph.bindTensorShapeDimensions(feed_dict)

            out_dims = TensorShape(in_phs[-1].shape.dims)
            out_dims.dims[axis] = out_c_dim
            check_symbol_table = {}
            for idx in range(concat_out.shape.rank):
                c_out_dim = concat_out.shape.getDim(idx)
                out_dim = out_dims.getDim(idx)
                if isinstance(out_dim, sympy.Symbol) and \
                   isinstance(c_out_dim, int):
                    if out_dim not in check_symbol_table.keys():
                        check_symbol_table[out_dim] = c_out_dim
                        out_dim = c_out_dim
                    else:
                        assert c_out_dim == check_symbol_table[out_dim]
                print('    CouGr dim[{}]:   {}'.format(idx, c_out_dim))
                print('    Correct dim[{}]: {}'.format(idx, out_dim))
                assert (sympy.simplify(c_out_dim - out_dim) == 0), \
                    'Concat dim[{}] incorrect!\n  Expecting:  {}\n' \
                    '  Calculated: {}'.format(idx, out_dim, c_out_dim)
        reset_symbols()


if __name__ == "__main__":
    test_concat_op()


