

import catamount
from catamount.ops.base_op import *

def test_op():
    op_name = 'test_op'
    my_op = Op(name=op_name)
    assert(my_op.name == op_name)
