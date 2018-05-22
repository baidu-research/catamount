from .base_op import Op


class PlaceholderOp(Op):
    def __init__(self, name):
        super(PlaceholderOp, self).__init__(name)

    def bindTensorShapeName(self, dim_index, shape_symbol):
        assert len(self._outputs) == 1
        out_shape = self._outputs[0].shape.setDimName(dim_index, shape_symbol)

    def propagateShapes(self):
        # Placeholders have no inputs to propagate
        pass

    def calcAlgFlops(self):
        # Placeholders have no Flops
        return 0
