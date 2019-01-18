import sympy

import catamount
from catamount.api import utils
from catamount.graph import Graph
from catamount.tensors.tensor_shape import Dimension, TensorShape

from catamount.tests.utils.helpers import *


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
                        append_dim_sym = utils.getIntSymbolFromString(dim_name)
                    else:
                        append_dim_sym = dim
                    in_ph_dims.append(append_dim_sym)
                    if idx == axis:
                        append_dim = Dimension(None)
                        append_dim.setSymbolOrName(append_dim_sym)
                        out_c_dim += append_dim
                feed_dict[in_ph.name] = in_ph_dims
            print('    Feed dict: {}'.format(feed_dict))

            graph.bindShapesAndPropagate(feed_dict)

            out_dims = TensorShape(in_phs[-1].shape.dims)
            out_dims.dims[axis] = out_c_dim
            check_symbol_table = {}
            for idx in range(concat_out.shape.rank):
                c_out_dim = concat_out.shape.getDimension(idx).symbol
                out_dim = out_dims.getDimension(idx).symbol
                if isinstance(out_dim, sympy.Symbol) and \
                   isinstance(c_out_dim, int):
                    if out_dim not in check_symbol_table.keys():
                        check_symbol_table[out_dim] = c_out_dim
                        out_dim = c_out_dim
                    else:
                        assert c_out_dim == check_symbol_table[out_dim]
                print('    Catamount dim[{}]:   {}'.format(idx, c_out_dim))
                print('    Correct dim[{}]: {}'.format(idx, out_dim))
                assert (sympy.simplify(c_out_dim - out_dim) == 0), \
                    'Concat dim[{}] incorrect!\n  Expecting:  {}\n' \
                    '  Calculated: {}'.format(idx, out_dim, c_out_dim)
        reset_symbols()


if __name__ == "__main__":
    test_concat_op()


