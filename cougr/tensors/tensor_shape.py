import re
import sympy


def as_dimension(value):
    if isinstance(value, Dimension):
        return value
    elif isinstance(value, int):
        return Dimension(value)
    else:
        raise TypeError('Cannot convert value of type {} to Dimension'
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
        else:
            raise TypeError('Unknown Dimension type {}'.format(type(value)))
        self._symbol = None

    def __str__(self):
        to_return = self.symbol
        if to_return is None:
            to_return = '?'
        return str(to_return)

    def setSymbolOrName(self, symbol_or_name):
        if isinstance(symbol_or_name, str):
            self.setSymbolName(symbol_or_name)
        elif isinstance(symbol_or_name, int):
            self._value = symbol_or_name         
        elif isinstance(symbol_or_name, sympy.Symbol):
            self._symbol = symbol_or_name
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
        return True

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
            self._dims = []
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
            elif self.dims[idx] is None or other.dims[idx] is None:
                print('WARN: May need to check if dimension '
                      'symbols are the same')
        return True

    def isValid(self):
        if not isinstance(self._dims, list):
            print('WARN: TensorShape dims must be a list (got {})!'
                  .format(type(self._dims)))
            return False
        for dim in self._dims:
            if dim is not None and not isinstance(dim, Dimension):
                print('WARN: Shape dim {} is not valid type'.format(dim))
        return True

    @property
    def dims(self):
        '''Returns a list of dimensions.
        '''
        return self._dims

    @property
    def rank(self):
        return len(self._dims)

    def associateTensor(self, tensor):
        self._tensor = tensor

    def setDimension(self, dim_index, dim_symbol_or_name):
        assert len(self._dims) > dim_index
        self._dims[dim_index].setSymbolOrName(dim_symbol_or_name)

    def getSymbolName(self, dim_index):
        assert self._tensor is not None
        assert dim_index < len(self._dims)
        return '{}::dim_{}'.format(self._tensor.name, dim_index)

    def getDim(self, idx):
        assert(idx < len(self._dims))
        to_return = self._dims[idx]
        if to_return.symbol is None:
            to_return.setSymbolName(self.getSymbolName(idx))
        assert to_return.symbol is not None
        return to_return.symbol

    def numElements(self):
        num_elts = Dimension(1)
        for idx, dim in enumerate(self._dims):
            if dim.symbol is None:
                dim.setSymbolName(self.getSymbolName(idx))
            num_elts *= dim
        assert num_elts.symbol is not None
        return num_elts.symbol
