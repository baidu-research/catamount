import numpy as np
import sympy


def as_dimension(value):
    if isinstance(value, Dimension):
        return value
    elif isinstance(value, int):
        return Dimension(value)
    elif isinstance(value, (sympy.Symbol, sympy.Expr)):
        to_return = Dimension(None)
        to_return.setSymbolOrName(value)
        return to_return
    else:
        raise TypeError('Cannot convert value of type {} to Dimension'
                        .format(type(value)))

def as_tensor_shape(value):
    if isinstance(value, TensorShape):
        return value
    elif isinstance(value, list) or isinstance(value, np.ndarray):
        list_dims = [as_dimension(dim) for dim in value]
        return TensorShape(list_dims)
    else:
        raise TypeError(
            'Cannot convert value of type {} to TensorShape'
            .format(type(value)))


class Dimension(object):
    ''' Represents a dimension of a `TensorShape`.
    In CouGr, a dimension has a name (symbol) and value (int), either of which
    can be None at any time.

    A value of None indicates that the Dimension has not been bound to an
    integer value. In that case, the symbol will be handled instead.
    '''
    def __init__(self, value=None):
        if value is None or isinstance(value, int):
            self._value = value
            self._symbol = None
        elif isinstance(value, Dimension):
            self._value = value._value
            self._symbol = value._symbol
        else:
            raise TypeError('Unknown Dimension type {}'.format(type(value)))

    def __str__(self):
        if self._value is None:
            to_return = '?'
        else:
            to_return = str(self._value)
        if self._symbol is not None:
            to_return = '{} "{}"'.format(to_return, self._symbol)
        return 'Dimension({})'.format(to_return)

    def setSymbolOrName(self, symbol_or_name):
        if isinstance(symbol_or_name, str):
            self.setSymbolName(symbol_or_name)
        elif isinstance(symbol_or_name, int):
            self._value = symbol_or_name         
        elif isinstance(symbol_or_name, (sympy.Symbol, sympy.Expr)):
            self._symbol = symbol_or_name
        elif isinstance(symbol_or_name, Dimension):
            # Need to copy self._value and self._symbol if they are not None
            if self._value is None:
                self._value = symbol_or_name.value
            else:
                assert symbol_or_name.value is None or \
                       self._value == symbol_or_name.value
            # Always propagate symbols
            self._symbol = symbol_or_name._symbol
        else:
            raise TypeError('Unknown symbol type {}'
                            .format(type(symbol_or_name)))

    def setSymbolName(self, symbol_name):
        assert(isinstance(symbol_name, str))
        self._symbol = sympy.Symbol(symbol_name)

    @property
    def value(self):
        return self._value

    @property
    def symbol(self):
        # Always select the value to which the dimension is bound before
        # deciding to return the symbol
        if self._value is not None:
            return self._value
        return self._symbol

    def __eq__(self, other):
        ''' Dimension equality is reflexive and symmetric, but not transitive
        '''
        other = as_dimension(other)
        if self._value is None or other._value is None or \
            self._value == other._value:
            return True
        return False

    def __iadd__(self, other):
        assert isinstance(other, Dimension)
        if self.symbol is not None and other.symbol is not None:
            self._symbol = self.symbol + other.symbol
        if self._value is None or other._value is None:
            self._value = None
        else:
            self._value += other._value
        return self

    def __mul__(self, other):
        assert isinstance(other, Dimension)
        to_return = Dimension()
        if self._value is None or other._value is None:
            to_return._value = None
        else:
            to_return._value = self._value * other._value
        if self.symbol is not None and other.symbol is not None:
            to_return._symbol = self.symbol * other.symbol
        return to_return

    def canBroadcastTogether(self, other):
        # CouGr implements broadcasting rules according to Numpy here:
        # https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
        # Two dimensions can broadcast together if their values are the
        # same or if either is 1 or None
        return (self == other or \
                self.value == None or self.value == 1 or \
                other.value == None or other.value == 1)

    def getBroadcastDimension(self, other):
        assert isinstance(other, Dimension)
        # CouGr implements broadcasting rules according to Numpy here:
        # https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
        # Two dimensions can broadcast together if their values are the
        # same or if either is 1 or None
        if self._value is None or other._value is None:
            new_value = None
        elif self._value == 1:
            new_value = other._value
        elif other._value == 1:
            new_value = self._value
        else:
            assert self._value == other._value
            new_value = self._value

        # Symbols should propagate with a warning if they do not match
        if self._symbol is None and other._symbol is None:
            new_symbol = None
        elif self._symbol is None:
            new_symbol = other._symbol
        elif other._symbol is None:
            new_symbol = self._symbol
        else:
            if self._symbol != other._symbol:
                print('WARN: Dimension symbols do not match: {} != {}'.format(
                      self._symbol, other._symbol))
            new_symbol = self._symbol
        new_dim = Dimension(new_value)
        if new_symbol is not None:
            new_dim.setSymbolOrName(new_symbol)
        return new_dim


class TensorShape(object):
    '''Represents the shape of a `Tensor`.
    A `TensorShape` represents a possibly-partial shape specification for a
    `Tensor`.
    '''
    def __init__(self, dims):
        '''Creates a new TensorShape with the given dimensions.

        Args:
            dims: A list of dimensions (int), or None if the shape is unspecified.

        Raises:
            TypeError: If dims cannot be converted to a list of dimensions.
        '''
        self._tensor = None
        if dims is None:
            self._dims = None
        elif isinstance(dims, int):
            self._dims = [Dimension(dims)]
        elif isinstance(dims, list):
            self._dims = []
            for dim in dims:
                assert(isinstance(dim, int) or \
                       isinstance(dim, Dimension) or \
                       dim is None)
                self._dims.append(Dimension(dim))
        elif isinstance(dims, TensorShape):
            self._dims = dims.dims
        else:
            raise TypeError('Unknown TensorShape type {}'.format(type(dims)))

    def __repr__(self):
        return 'TensorShape({})'.format(self._dims)

    def __str__(self):
        if self._dims is None:
            return '?'
        return '({})'.format(', '.join(str(d) for d in self._dims))

    def __eq__(self, other):
        if self.rank == 0 and other.rank == 0:
            return True
        if self.rank != other.rank:
            return False
        for idx in range(self.rank):
            if self.dims[idx] is not None and other.dims[idx] is not None:
                if self.dims[idx] != other.dims[idx]:
                    return False
            elif self.dims[idx] is None or other.dims[idx] is None:
                print('WARN: May need to check if dimension '
                      'symbols are the same')
        return True

    def isValid(self):
        if self._dims is None:
            return True
        if not isinstance(self._dims, list):
            print('WARN: TensorShape dims must be a list (got {})!'
                  .format(type(self._dims)))
            return False
        for dim in self._dims:
            if dim is not None and not isinstance(dim, Dimension):
                print('WARN: Shape dim {} is not valid type'.format(dim))
        return True

    def isUnknown(self):
        return self._dims is None

    def isScalar(self):
        if self._dims is None:
            return False
        return len(self._dims) == 0

    def isFullyDefined(self):
        # Return whether the shape has all dimension values set
        if self._dims is None:
            return False
        for dim in self._dims:
            if dim.value is None:
                return False
        return True

    def isFullySymbolic(self):
        # Return whether the shape has all dimension values or symbols set
        if self._dims is None:
            return False
        for dim in self._dims:
            if dim.symbol is None:
                return False
        return True

    @property
    def dims(self):
        '''Returns a list of dimensions.
        '''
        return self._dims

    @property
    def rank(self):
        if self._dims is None:
            # TODO (Joel): This might also work with -1? Finalize choice later
            return None
        return len(self._dims)

    def canBroadcastTogether(self, other):
        ''' Returns whether this TensorShape can be broadcast to other
        TensorShape or vice-versa. This check is reflexive and symmetric,
        but not transitive.
        '''
        assert isinstance(other, TensorShape)
        # CouGr implements broadcasting rules according to Numpy here:
        # https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
        # Get first and second shape, longest first
        if self.rank < other.rank:
            first_shape = list(other.dims)
            second_shape = list(self._dims)
        else:
            first_shape = list(self._dims)
            second_shape = list(other.dims)
        # Iterate from trailing dimensions to early dimensions
        for idx in range(len(second_shape) - 1, -1, -1):
            if not first_shape[idx].canBroadcastTogether(second_shape[idx]):
                return False
        return True

    def getBroadcastShape(self, other):
        assert isinstance(other, TensorShape)
        if self.rank < other.rank:
            first_shape = list(other.dims)
            second_shape = list(self._dims)
        else:
            first_shape = list(self._dims)
            second_shape = list(other.dims)
        # Extend second shape to first shape length
        while len(second_shape) < len(first_shape):
            second_shape.insert(0, Dimension(1))
        # Now perform broadcast
        bcast_shape_list = []
        for idx in range(len(first_shape) - 1, -1, -1):
            assert first_shape[idx].canBroadcastTogether(second_shape[idx])
            bcast_shape_list.insert(0,
                first_shape[idx].getBroadcastDimension(second_shape[idx]))
        bcast_shape = TensorShape(bcast_shape_list)
        # print('Bcast: {} x {} => {}'.format(self, other, bcast_shape))
        return bcast_shape

    def mergeShape(self, other):
        other = as_tensor_shape(other)
        # TODO (Joel): These checks will not work when a TensorShape
        # is None! Fix later
        assert self.rank == other.rank, \
            'Ranks: {} != {}'.format(self, other)
        assert self._dims is not None
        assert other.dims is not None
        # Now perform merging
        for idx, dim in enumerate(other.dims):
            if self._dims[idx].value is None:
               self.setDimension(idx, dim)
            else:
               assert self._dims[idx] == dim, \
                   'Dimension mismatch in Tensor {}: self[{}] {}, other {}' \
                   .format(self._tensor, idx, self._dims[idx], dim)
               if self._dims[idx]._symbol is None:
                   self.setDimension(idx, dim)

    def associateTensor(self, tensor):
        self._tensor = tensor

    def setDimension(self, dim_index, dim_symbol_or_name):
        if self._dims is None:
            # Assume that the caller has right to extend dimensions
            print('WARN: Adding dimensions to None TensorShape')
            self._dims = [Dimension(None)] * (dim_index + 1)
        assert len(self._dims) > dim_index, \
            'Trying to set dim {} outside bounds {} to {}' \
            .format(dim_index, len(self._dims), dim_symbol_or_name)
        self._dims[dim_index].setSymbolOrName(dim_symbol_or_name)

    def getSymbolName(self, dim_index):
        assert self._tensor is not None
        assert self._dims is None or dim_index < len(self._dims)
        return '{}::dim_{}'.format(self._tensor.name, dim_index)

    def getDimension(self, idx):
        assert(idx < len(self._dims))
        to_return = Dimension(self._dims[idx])
        if to_return.symbol is None:
            to_return.setSymbolName(self.getSymbolName(idx))
        return to_return

    def asList(self):
        assert self.isFullyDefined()
        to_return = []
        for dim in self._dims:
            to_return.append(dim.value)
        return to_return

    def numElements(self):
        if self._dims is None:
            # Unknown dimensionality... return '?'
            return sympy.Symbol(self.getSymbolName('?'))
        num_elts = Dimension(1)
        for idx, dim in enumerate(self._dims):
            if dim.value is None:
                if dim.symbol is None:
                    dim = Dimension(None)
                    dim.setSymbolName(self.getSymbolName(idx))
            num_elts *= dim
        assert num_elts.symbol is not None
        return num_elts.symbol
