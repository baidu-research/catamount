import sympy

import catamount
from catamount.graph import Graph

from catamount.tests.utils.helpers import *


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

    for second_input in [False, True]:
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
                if second_input:
                    const_dim = []
                    if isinstance(axes, list):
                        const_dim.append(len(axes))
                    axes = constant('const', const_dim, axes)
                reduced_a = reduce('reduce', 'sum', out_dims, in_a_ph,
                                   axes=axes)

                algorithmic_flops = graph.calcAlgFlops()

                correct_alg_flops = get_correct_alg_flops()
                subs_table = get_subs_table()
                correct_alg_flops = correct_alg_flops.subs(subs_table)
                print('    Catamount:   {}'.format(algorithmic_flops))
                print('    Correct: {}'.format(correct_alg_flops))
                diff_flops = algorithmic_flops - correct_alg_flops
                assert sympy.simplify(diff_flops) == 0,\
                    'Alg flops incorrect!\n  Expecting:  {}\n  ' \
                    'Calculated: {}' \
                    .format(correct_alg_flops, algorithmic_flops)
            # TODO: Bind Nones and check outputs
            reset_symbols()


if __name__ == "__main__":
    test_reduce_op()


