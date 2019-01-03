import sympy
import numpy as np

import catamount
from catamount.graph import Graph

from catamount.tests.utils.helpers import *


def test_dynamic_stitch_op():
    ''' Specify graphs with DynamicStitch operations and make sure they behave
    as desired.
    '''

    combos = [ # Speech examples
               ([[0, 1], 0], # indices
                [[5, 6], 7], # data
                [7, 6]),
               ([[0, 1, 2], 0], # indices
                [None, 1], # data
                None), # merged
               # TF Docs example
               ([6, [4, 1], [[5, 2], [0, 3]]], # indices
                [[61, 62], [[41, 42], [11, 12]], [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]], # data
                [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42], [51, 52], [61, 62]]) # merged
             ]

    for combo in combos:
        graph = Graph()
        with graph.asDefault():
            indices, data, merged = combo
            ind_consts = []
            data_consts = []
            for idx in range(len(indices)):
                ind_const_name = 'int_const_{}'.format(idx)
                shape = list(np.array(indices[idx]).shape)
                ind_consts.append(
                    constant(ind_const_name, shape, indices[idx]))

                data_const_name = 'data_const_{}'.format(idx)
                shape = list(np.array(data[idx]).shape)
                data_consts.append(
                    constant(data_const_name, shape, data[idx]))
            out_tensor = dynamic_stitch('dyn_stitch', None, ind_consts,
                                        data_consts)

            bind_dict = {}
            graph.bindShapesAndPropagate(bind_dict,
                                         warn_if_ill_defined=True)

            # Check that out_tensor shape and value is correct!
            out_value = out_tensor.value
            assert np.array_equal(out_value, merged)
            print('Out tensor: {}, correct: {}'.format(out_tensor, merged))
        reset_symbols()


if __name__ == "__main__":
    test_dynamic_stitch_op()


