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

    def __str__(self):
        value_to_print = self._value
        if isinstance(self._value, list):
            if len(self._value) > 10:
                value_to_print = '['
                for i in range(10):
                    value_to_print += '{}, '.format(self._value[i])
                value_to_print += '...]'
        return 'Tensor(name: {}, shape: {}, value: {})' \
               .format(self._name, self._shape, value_to_print)

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

    def setValue(self, value):
        supported_python_types = ( bool, int, float, sympy.Symbol,
                                   sympy.Expr, str, np.int64, np.str_ )
        np_string_types = ( 'U', 'S' )
        if DataType.isNumber(self._dtype) or DataType.isString(self._dtype):
            if self._shape.rank == 0:
                if isinstance(value, list) or isinstance(value, np.ndarray):
                    assert len(value) == 1
                    value = value[0]
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
                assert list(value.shape) == self._shape.asList(), \
                    '{}:\nShape mismatch. Value {}, shapes {} != {}' \
                    .format(self, value, list(value.shape),
                            self._shape.asList())
        else:
            raise NotImplementedError('Yet unsupported dtype: {}'
                                      .format(self._dtype))
        self._value = value
