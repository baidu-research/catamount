from .base_op import Op


class PlaceholderOp(Op):
    def __init__(self, name):
        super(PlaceholderOp, self).__init__(name)

    def bindTensorShapeDimension(self, dim_index, dim_name_or_symbol):
        assert len(self._outputs) == 1
        self._outputs[0].shape.setDimension(dim_index, dim_name_or_symbol)

    def propagateShapes(self):
        # Placeholders have no inputs to propagate
        pass

    def calcAlgFlops(self):
        # Placeholders have no Flops
        return 0
