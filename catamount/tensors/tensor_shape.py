import numpy as np
import sympy

from ..api import utils


def as_dimension(value):
    if isinstance(value, Dimension):
        return value
    elif isinstance(value, (int, np.int64)):
        return Dimension(int(value))
    elif isinstance(value, (str, sympy.Symbol, sympy.Expr)):
        if isinstance(value, str):
            value = utils.getIntSymbolFromString(value)
        to_return = Dimension(None)
        to_return.setSymbolOrName(value.simplify())
        return to_return
    elif isinstance(value, np.float64):
        assert value.is_integer()
        return Dimension(int(value))
    elif value is None:
        return Dimension(None)
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
    In Catamount, a dimension has a name (symbol) and value (int), either of which
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

    def setSymbolOrName(self, symbol_or_name, make_symbolic=False):
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
            # Always propagate symbols unless new symbol is None
            if symbol_or_name._symbol is not None:
                self._symbol = symbol_or_name._symbol
        elif symbol_or_name is None:
            print('WARN: Trying to set symbol or name to None in {}'
                  .format(self))
        else:
            raise TypeError('Unknown symbol type {}'
                            .format(type(symbol_or_name)))

        if make_symbolic:
            # When using symbolic-only propagation, clear values for
            # dimensions that have a valid symbolic value
            if self._symbol is not None and self._value is not None:
                assert isinstance(self._symbol, sympy.Expr)
                self._value = None

    def setSymbolName(self, symbol_name):
        assert(isinstance(symbol_name, str))
        # Dimensions have integer types, so specify that this symbol
        # represents an integer
        self._symbol = utils.getIntSymbolFromString(symbol_name)

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
        other = as_dimension(other)
        my_new_dim = Dimension()
        if self._value is None or other._value is None:
            # If either values is None, then cannot calculate an integer
            # dimension, so self._value must be set to None
            my_new_dim._value = None
        else:
            my_new_dim._value = self._value + other._value
            assert isinstance(my_new_dim._value, int)
        if self._symbol is not None:
            if other._symbol is not None:
                my_new_dim._symbol = self._symbol + other._symbol
            else:
                assert other._value is not None
                my_new_dim._symbol = self._symbol + other._value
        else:
            assert self._value is not None
            if other._symbol is not None:
                my_new_dim._symbol = self._value + other._symbol
            else:
                # Cannot set symbol, because only have values
                pass
        if my_new_dim._symbol is not None:
            assert isinstance(my_new_dim._symbol, sympy.Expr)
            my_new_dim._symbol = my_new_dim._symbol.simplify()
        self._value = my_new_dim._value
        self._symbol = my_new_dim._symbol
        return self

    def __mul__(self, other):
        other = as_dimension(other)
        to_return = Dimension()
        if self._value is None or other._value is None:
            to_return._value = None
        else:
            to_return._value = self._value * other._value
            assert isinstance(self._value, int)
        if self._symbol is not None:
            if other._symbol is not None:
                to_return._symbol = self._symbol * other._symbol
            else:
                assert other._value is not None
                to_return._symbol = self._symbol * other._value
        else:
            assert self._value is not None
            if other._symbol is not None:
                to_return._symbol = self._value * other._symbol
            else:
                # Cannot set symbol, because only have values
                pass
        if to_return._symbol is not None:
            assert isinstance(to_return._symbol, sympy.Expr)
            to_return._symbol = to_return._symbol.simplify()
        return to_return

    def __floordiv__(self, other):
        # TODO (Joel): Loosen this as appropriate
        assert isinstance(other, int)
        to_return = Dimension()
        if self._value is not None:
            to_return._value = self._value // other
        if self._symbol is not None:
            to_return._symbol = (self._symbol // other).simplify()
        return to_return

    def canBroadcastTogether(self, other):
        # Catamount implements broadcasting rules according to Numpy here:
        # https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
        # Two dimensions can broadcast together if their values are the
        # same or if either is 1 or None
        return (self == other or \
                self.value == None or self.value == 1 or \
                other.value == None or other.value == 1)

    def getBroadcastDimension(self, other):
        assert isinstance(other, Dimension)
        # Catamount implements broadcasting rules according to Numpy here:
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
                       isinstance(dim, sympy.Expr) or \
                       dim is None)
                self._dims.append(as_dimension(dim))
        elif isinstance(dims, TensorShape):
            self._dims = []
            for dim in dims.dims:
                self._dims.append(Dimension(dim))
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
        if self.rank is None:
            return True
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
        if len(self._dims) == 0:
            return True
        else:
            for dim in self._dims:
                if dim.value != 1:
                    return False
            return True

    def isFullyNumeric(self):
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
        # Catamount implements broadcasting rules according to Numpy here:
        # https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
        # Get first and second shape, longest first
        if self.rank < other.rank:
            first_shape = list(other.dims)
            second_shape = list(self._dims)
        else:
            first_shape = list(self._dims)
            second_shape = list(other.dims)
        # Extend the shorter shape list with broadcastable dimensions
        while len(second_shape) < len(first_shape):
            second_shape.insert(0, Dimension(1))
        # Iterate from trailing dimensions to early dimensions
        for idx in range(len(first_shape) - 1, -1, -1):
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

    def mergeShape(self, other, make_symbolic=False):
        other = as_tensor_shape(other)
        assert self.rank is None or self.rank == other.rank or \
            self.canBroadcastTogether(other) or \
            (self.isScalar() and other.isScalar()), \
            'Ranks: {} != {}'.format(self, other)
        assert other.dims is not None
        if self.isScalar() and self.rank == 0:
            if other.rank == 1 and other.isScalar():
                # Keep scalars as rank = 0
                return
        # Now perform merging
        if self.rank is None:
            self._dims = [Dimension(None) for x in range(other.rank)]
        for idx, dim in enumerate(other.dims):
            if self._dims[idx].value is None:
               self.setDimension(idx, dim, make_symbolic=make_symbolic)
            else:
               assert self._dims[idx] == dim, \
                   'Dimension mismatch in Tensor {}: self[{}] {}, other {}' \
                   .format(self._tensor, idx, self._dims[idx], dim)
               if self._dims[idx]._symbol is None:
                   self.setDimension(idx, dim, make_symbolic=make_symbolic)

    def associateTensor(self, tensor):
        self._tensor = tensor

    def setDimension(self, dim_index, dim_symbol_or_name,
                     make_symbolic=False):
        ''' Set the TensorShape's specified dimension (dim_index) to the
            specified symbol or name.

            Args:
              dim_index (int): The dimension index to change
              dim_symbol_or_name: The int, string, or symbol to change the
                  dimension to. Different types are resolved appropriately
              make_symbolic (bool): Whether to clear the value of the
                  dimension if the symbol is specified.
        '''
        if self._dims is None:
            # Assume that the caller has right to extend dimensions
            print('WARN: Adding dimensions to None TensorShape')
            self._dims = [Dimension(None)] * (dim_index + 1)
        assert len(self._dims) > dim_index, \
            'Trying to set dim {} outside bounds {} to {}' \
            .format(dim_index, len(self._dims), dim_symbol_or_name)
        self._dims[dim_index].setSymbolOrName(dim_symbol_or_name,
                                              make_symbolic=make_symbolic)

    def getSymbolName(self, dim_index):
        assert self._tensor is not None
        assert self._dims is None or dim_index < len(self._dims)
        return '{}::dim_{}'.format(self._tensor.name, dim_index)

    def getDimension(self, idx):
        assert idx < len(self._dims), \
               'Dimension {} out-of-bounds for Tensor {}' \
               .format(idx, self._tensor)
        to_return = Dimension(self._dims[idx])
        if to_return.symbol is None:
            to_return.setSymbolName(self.getSymbolName(idx))
        return to_return

    def asList(self):
        assert not self.isUnknown()
        to_return = []
        for dim in self._dims:
            to_return.append(dim.symbol)
        return to_return

    def numElements(self):
        if self._dims is None:
            # Unknown dimensionality... return '?'. Type is integer
            return utils.getIntSymbolFromString(self.getSymbolName('?'))
        num_elts = Dimension(1)
        for idx, dim in enumerate(self._dims):
            if dim.value is None:
                if dim.symbol is None:
                    dim = Dimension(None)
                    dim.setSymbolName(self.getSymbolName(idx))
            num_elts *= dim
        assert num_elts.symbol is not None
        return num_elts.symbol
