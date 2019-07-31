import numpy as np
import sympy

from enum import Enum, unique
from .tensor_shape import TensorShape, Dimension


@unique
class DataType(Enum):
    bool = 0
    int8 = 1
    int16 = 2
    int32 = 3
    int64 = 4
    uint8 = 5
    uint16 = 6
    uint32 = 7
    uint64 = 8
    float16 = 9
    float32 = 10
    float64 = 11

    string = 12

    int8_ref = 21
    int16_ref = 22
    int32_ref = 23
    int64_ref = 24
    float16_ref = 25
    float32_ref = 26
    float64_ref = 27

    def isNumber(type):
        return (type == DataType.bool) or \
               (type == DataType.int8) or \
               (type == DataType.int16) or \
               (type == DataType.int32) or \
               (type == DataType.int64) or \
               (type == DataType.uint8) or \
               (type == DataType.uint16) or \
               (type == DataType.uint32) or \
               (type == DataType.uint64) or \
               (type == DataType.float16) or \
               (type == DataType.float32) or \
               (type == DataType.float64)

    def isString(type):
        return (type == DataType.string)

    def sizeof(type):
        sizeof = {
                   DataType.bool: 1,
                   DataType.int8: 1,
                   DataType.int16: 2,
                   DataType.int32: 4,
                   DataType.int64: 8,
                   DataType.uint8: 1,
                   DataType.uint16: 2,
                   DataType.uint32: 4,
                   DataType.uint64: 8,
                   DataType.float16: 2,
                   DataType.float32: 4,
                   DataType.float64: 8,
                 }
        return sizeof[type]

    def numpytype(type):
        numpytype = {
            DataType.bool: np.bool_,
            DataType.int8: np.int8,
            DataType.int16: np.int16,
            DataType.int32: np.int32,
            DataType.int64: np.int64,
            DataType.uint8: np.uint8,
            DataType.uint16: np.uint16,
            DataType.uint32: np.uint32,
            DataType.uint64: np.uint64,
            DataType.float16: np.float16,
            DataType.float32: np.float32,
            DataType.float64: np.float64,
        }
        return numpytype[type]

    def cast(in_val, out_dtype):
        if isinstance(in_val, np.ndarray):
            if in_val.dtype == object:
                print('WARN: Should cast Numpy array in_val {} to type {}'
                      .format(in_val, out_dtype))
                out_val = in_val
            else:
                out_val = in_val.astype(DataType.numpytype(out_dtype))
        else:
            if isinstance(in_val, sympy.Expr):
                print('WARN: Should cast Sympy in_val {} to type {}'
                      .format(in_val, out_dtype))
                out_val = in_val
            else:
                if out_dtype == DataType.float32:
                    out_val = float(in_val)
                else:
                    print('in_val: {}'.format(in_val))
                    print('Out type: {}, Numpy: {}'
                          .format(out_dtype, DataType.numpytype(out_dtype)))
                    raise NotImplementedError('DataType non-Numpy cast')
        return out_val


class Tensor:
    def __init__(self, name, shape, dtype=DataType.float32):
        self._name = name
        self._shape = shape
        self._shape.associateTensor(self)
        self._dtype = dtype
        self._producer = None
        self._consumers = {}
        self._value = None

    @property
    def name(self):
        return self._name

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def producer(self):
        return self._producer

    @property
    def consumers(self):
        return self._consumers

    @property
    def value(self):
        return self._value

    @property
    def size(self):
        if self._dtype is None:
            # Unknown DataType: Skip
            return 0
        if self._dtype.isString():
            # TODO (Joel): For now, assume strings are ignored, since they
            # are not operated on by any compute graph ops. Maybe later, we
            # will need to count these in the bytes...?
            return 0
        return DataType.sizeof(self._dtype) * self._shape.numElements()

    def __str__(self):
        return 'Tensor(name: {}, shape: {}, value: {})' \
               .format(self._name, self._shape, self._value)

    def isValid(self):
        # Valid tensors have a valid TensorShape
        if type(self._shape) is not TensorShape or not self._shape.isValid():
            print('WARN: Invalid shape for tensor {}'.format(self._name))
            return False
        # Valid values have correct shape
        if self._value is not None:
            np_shape = np.shape(self._value)
            for idx, dim in enumerate(np_shape):
                shape_dim = self.shape.getDimension(idx).value
                if dim != shape_dim:
                    print('WARN: Value dim[{}] = {} mismatch with shape {} ' \
                          'in tensor:\n {}'.format(idx, dim, shape_dim, self))
                    return False
        return True

    def setProducer(self, op):
        assert self._producer is None
        self._producer = op

    def addConsumer(self, op):
        if op.name in self._consumers.keys():
            assert self._consumers[op.name] == op
            return
        self._consumers[op.name] = op

    def hasConsumers(self):
        return len(self._consumers.keys()) > 0

    def removeConsumer(self, op):
        cons_op = self._consumers.pop(op.name, None)
        assert cons_op is None or cons_op == op

    def getFreeSymbols(self):
        to_return = set()
        for dim in self._shape.dims:
            if dim._symbol is not None:
                to_return.update(dim._symbol.free_symbols)
        return to_return

    def mergeShape(self, other, make_symbolic=False):
        self.shape.mergeShape(other, make_symbolic=make_symbolic)
        # If the new shape is under-specified, the tensor cannot track
        # a fully-specified value. In that case, clear the value.
        if not self.shape.isFullyNumeric() and self._value is not None:
            self._value = None

    def setValue(self, value):
        supported_python_types = ( bool, int, float, sympy.Symbol,
                                   sympy.Expr, str, np.int64, np.int32,
                                   np.str_, np.bool_ )
        np_string_types = ( 'U', 'S' )
        if DataType.isNumber(self._dtype) or DataType.isString(self._dtype):
            if self._shape.rank == 0:
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                if isinstance(value, list):
                    assert len(value) == 1
                    value = value[0]
                if isinstance(value, (sympy.boolalg.BooleanTrue,
                                      sympy.boolalg.BooleanFalse)):
                    value = bool(value)
                assert isinstance(value, supported_python_types), \
                    'Tensor {} setting value to {} with type {}' \
                    .format(self, value, type(value))
            else:
                if isinstance(value, supported_python_types):
                    value = np.array([value])
                elif isinstance(value, list):
                    value = np.array(value)
                assert isinstance(value, np.ndarray), \
                    'Tensor {} setting value to {} with type {}' \
                    .format(self, value, type(value))
                assert value.dtype in supported_python_types or \
                       value.dtype.kind in np_string_types, \
                    '{}:\nTrying to set value dtype to {}' \
                    .format(self, value.dtype)
                if list(value.shape) != self._shape.asList():
                    # Try to reshape if not already correct
                    try:
                        value = np.reshape(value, self._shape.asList())
                    except ValueError as err:
                        print('{}:\nShape mismatch. Value {}, shapes '\
                              '{} != {}'.format(self, value,
                                                list(value.shape),
                                                self._shape.asList()))
                        raise err
                assert list(value.shape) == self._shape.asList(), \
                    '{}:\nShape mismatch. Value {}, shapes {} != {}' \
                    .format(self, value, list(value.shape),
                            self._shape.asList())
        else:
            raise NotImplementedError('Yet unsupported dtype: {}'
                                      .format(self._dtype))
        self._value = value
