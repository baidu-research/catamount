import sympy

class TensorShape(object):
    """Represents the shape of a `Tensor`.

    A `TensorShape` represents a possibly-partial shape specification for a
    `Tensor`. It may be one of the following:

    """
    def __init__(self, dims):
        """Creates a new TensorShape with the given dimensions.

        Args:
            dims: A list of dimensions (int), or None if the shape is unspecified.

        Raises:
            TypeError: If dims cannot be converted to a list of dimensions.
        """
        if dims is None:
            self._dims = []
        elif isinstance(dims, int):
            self._dims = [dims]
        elif isinstance(dims, list):
            for dim in dims:
                assert(isinstance(dim, int) or dim is None)
            self._dims = dims
        elif isinstance(dims, TensorShape):
            self._dims = dims.dims
        else:
            raise TypeError("Unknown TensorShape type {}".format(type(dims)))

    def __repr__(self):
        return "TensorShape(%r)" % self._dims

    def __str__(self):
        # [_] TODO (Joel): FIXME!
        if self.ndims is None:
            return "<unknown>"
        elif self.ndims == 1:
            return "(%s,)" % self._dims[0]
        else:
            return "(%s)" % ", ".join(str(d) for d in self._dims)

    @property
    def dims(self):
        """Returns a list of Dimensions, or None if the shape is unspecified."""
        return self._dims

    def getDim(self, idx):
        assert(len(self._dims) > idx)
        to_return = self._dims[idx]
        if to_return is None:
            sym_name = 'dim_{}'.format(idx)
            to_return = sympy.Symbol(sym_name)
        return to_return

    def numElements(self):
        num_elts = 1
        for idx, dim in enumerate(self._dims):
            if dim is None:
                sym_name = 'dim_{}'.format(idx)
                dim_elts = sympy.Symbol(sym_name)
            else:
                dim_elts = dim
            num_elts *= dim_elts
        return num_elts
