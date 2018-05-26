import sympy

import cougr
from cougr.graph import Graph


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


def reduce(name, op_func, out_shape, input, axes=0):
    add_symbols(name, out_shape)

    global correct_alg_flops
    to_add_flops = 1
    for idx in range(input.shape.rank):
        in_dim = '{}::dim_{}'.format(input.name, idx)
        to_add_flops *= symbol_table[in_dim]
    correct_alg_flops += to_add_flops
    return cougr.reduce(name, op_func, out_shape, input, axes)


def test_reduce_op():
    ''' Specify graphs with reduce operations and make sure they behave as
    desired. 
    '''

    combos = [([None, None], 0),
              ([32, None], 0),
              ([None, 64], 0),
              ([32, 64], 0),
              ([None, None], 1),
              ([32, None], 1),
              ([None, 64], 1),
              ([32, 64], 1),
              ([None, None, None, None], [1, 2]),
              ([None, 3, 7, None], [0, 3]),
              ([None, 3, 7, None], [0, 2]),
              ([None, 3, 7, None], [1, 3]),
              ([None, 3, 7, None], [1, 2])]

    for combo in combos:
        graph = Graph()
        with graph.asDefault():
            dims, axes = combo
            if isinstance(axes, int):
                axes = [axes]
            out_dims = []
            for idx in range(len(dims)):
                if idx not in axes:
                    out_dims.append(dims[idx])
            print('Testing reduce with in dims {}, axes {}, out dims {}'
                  .format(dims, axes, out_dims))
            in_a_ph = placeholder('in_a', dims)
            reduced_a = reduce('reduce', 'sum', out_dims, in_a_ph, axes=axes)

            algorithmic_flops = graph.calcAlgFlops()

            global correct_alg_flops
            global subs_table
            correct_alg_flops = correct_alg_flops.subs(subs_table)
            print('    CouGr:   {}'.format(algorithmic_flops))
            print('    Correct: {}'.format(correct_alg_flops))
            assert sympy.simplify(algorithmic_flops - correct_alg_flops) == 0,\
                'Alg flops incorrect!\n  Expecting:  {}\n  Calculated: {}' \
                .format(correct_alg_flops, algorithmic_flops)
            # TODO: Bind Nones and check outputs
        reset_symbols()


if __name__ == "__main__":
    test_reduce_op()


