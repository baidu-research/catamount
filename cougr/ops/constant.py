from . import base_op


class NoOp(base_op.Op):
    ''' NoOps are special CouGr ops that fill in a place for non-functional
        compute graph ops, especially for those imported from different
        frameworks. For instance, Tensorflow Saver ops (Save, Restore) have
        no functional purpose in calculating compute graph outputs, but are
        there to load and save tensors.
    '''
    def __init__(self, name):
        super(NoOp, self).__init__(name)

class ConstOp(base_op.Op):
    def __init__(self, name):
        super(ConstOp, self).__init__(name)

