import re
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
        self._tensor = None
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
            raise TypeError('Unknown TensorShape type {}'.format(type(dims)))
        self._dim_names = [None for dim in self._dims]

    def __repr__(self):
        return 'TensorShape({})'.format(self._dims)

    def __str__(self):
        return '({})'.format(', '.join(str(d) for d in self._dims))

    def __eq__(self, other):
        if self.rank == 0 or other.rank == 0:
            return True
        if self.rank != other.rank:
            return False
        for idx in range(self.rank):
            if self.dims[idx] is not None and other.dims[idx] is not None:
                if self.dims[idx] != other.dims[idx]:
                    return False
            elif self.dims[idx] is None or self.dims[idx] is None:
                print('WARN: May need to check if dimension symbols are the same')
        return True

    @property
    def dims(self):
        """Returns a list of dimensions.
        """
        return self._dims

    @property
    def rank(self):
        return len(self._dims)

    def associateTensor(self, tensor):
        self._tensor = tensor

    def setDim(self, dim_index, dim_val_or_name):
        if type(dim_val_or_name) == sympy.Symbol:
            self.setDimName(dim_index, dim_val_or_name)
        else:
            if self._dims[dim_index] is None:
                self._dims[dim_index]
            else:
                assert dim_val_or_name == self._dims[dim_index]

    def setDimName(self, dim_index, shape_symbol):
        assert type(shape_symbol) == sympy.Symbol
        self._dim_names[dim_index] = shape_symbol

    def getDimSymbol(self, dim_index):
        assert self._tensor is not None
        assert dim_index < len(self._dims)
        # If the dimension name is not yet bound, return a template name
        if self._dim_names[dim_index] is None:
            sym_name = '{}::dim_{}'.format(self._tensor.name, dim_index)
            to_return = sympy.Symbol(sym_name)
        else:
            to_return = self._dim_names[dim_index]
        return to_return

    def getDim(self, idx):
        assert(idx < len(self._dims))
        to_return = self._dims[idx]
        if to_return is None:
            to_return = self.getDimSymbol(idx)
        return to_return

    def numElements(self):
        num_elts = 1
        for idx, dim in enumerate(self._dims):
            if dim is None:
                dim_elts = self.getDimSymbol(idx)
            else:
                dim_elts = dim
            num_elts *= dim_elts
        return num_elts
