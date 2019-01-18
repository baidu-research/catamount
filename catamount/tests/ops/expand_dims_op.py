import sympy

import catamount
from catamount.api import utils
from catamount.graph import Graph
from catamount.tensors.tensor_shape import Dimension, TensorShape

from catamount.tests.utils.helpers import *


def test_expanddims_op():
    ''' Specify graphs with ExpandDimsOps and make sure dimensions behave
    as desired.
    '''

    combos = [
              ([3], 0),
              ([None], 0),
              ([None, None], 0),
              ([None, None], 1),
              ([None, None], 2),
              ([None, None], -1),
              ([None, None], -2),
              ([None, None], -3),
             ]

    for combo in combos:
        graph = Graph()
        with graph.asDefault():
            ph_dims, expand_dim = combo
            if isinstance(ph_dims, list):
                ed_out_dims = list(ph_dims)
                insert_dim = expand_dim
                if insert_dim < 0:
                    insert_dim += len(ed_out_dims) + 1
                ed_out_dims.insert(insert_dim, 1)
            else:
                ed_out_dims = [ph_dims]
            print('Testing expand dims with in_dims {}, expand dim {} to {}'
                  .format(ph_dims, expand_dim, ed_out_dims))

            # Build model
            in_ph = placeholder('in', ph_dims)
            expanddims_out = expanddims('expanddims', ed_out_dims,
                                        in_ph, axis=expand_dim)

            assert graph.isValid()

            feed_dict = {}
            if isinstance(ph_dims, list):
                for idx in range(len(ph_dims)):
                    if ph_dims[idx] is None:
                        ph_dims[idx] = utils.getIntSymbolFromString(
                                           'in::dim_{}'.format(idx))
            else:
                if ph_dims is None:
                    ph_dims = utils.getIntSymbolFromString('in::dim_0')
            feed_dict['in'] = ph_dims
            print('    Feed dict: {}'.format(feed_dict))

            graph.bindShapesAndPropagate(feed_dict)
            assert expanddims_out.shape == TensorShape(ed_out_dims)
        reset_symbols()


if __name__ == "__main__":
    test_expanddims_op()


