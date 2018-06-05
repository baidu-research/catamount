import numpy as np

from enum import Enum, unique
from .tensor_shape import TensorShape, Dimension


@unique
class DataType(Enum):
    bool = 0
    int8 = 1
    int16 = 2
    int32 = 3
    int64 = 4
    uint16 = 5
    uint32 = 6
    uint64 = 7
    float16 = 8
    float32 = 9
    float64 = 10

    string = 11

    int8_ref = 21
    int16_ref = 22
    int32_ref = 23
    int64_ref = 24
    float16_ref = 25
    float32_ref = 26
    float64_ref = 27

    def isNumber(type):
        return (type == DataType.int8) or \
               (type == DataType.int16) or \
               (type == DataType.int32) or \
               (type == DataType.int64) or \
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
        return 'Tensor(name: {}, shape: {})'.format(self._name, self._shape)

    def isValid(self):
        # Valid tensors have a valid TensorShape
        if type(self._shape) is not TensorShape or not self._shape.isValid():
            print('WARN: Invalid shape for tensor {}'.format(self._name))
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
        # TODO (Joel): Re-write conditions to simplify these checks
        if DataType.isNumber(self._dtype):
            if self._shape.rank == 0:
                assert isinstance(value, int) or isinstance(value, float), \
                    'Tensor {} setting value to {} with type {}' \
                    .format(self, value, type(value))
            elif (self._shape.rank == 1 and self._shape.dims[0] == 1):
                if isinstance(value, int) or isinstance(value, float):
                    value = [value]
                assert isinstance(value, list), \
                    'Tensor {} setting value to {} with type {}' \
                    .format(self, value, type(value))
            else:
                assert isinstance(value, list), \
                    'Tensor {} setting value to {} with type {}' \
                    .format(self, value, type(value))
                # TODO (Joel): Make this check smarter
                assert len(value) == self._shape.numElements()
        elif DataType.isString(self._dtype):
            if self._shape.rank == 0:
                assert isinstance(value, str), \
                    'Tensor {} setting value to {} with type {}' \
                    .format(self, value, type(value))
            elif (self._shape.rank == 1 and self._shape.dims[0] == 1):
                if isinstance(value, str):
                    value = [value]
                assert isinstance(value, list), \
                    'Tensor {} setting value to {} with type {}' \
                    .format(self, value, type(value))
            else:
                assert isinstance(value, list), \
                    'Tensor {} setting value to {} with type {}' \
                    .format(self, value, type(value))
                # TODO (Joel): Make this check smarter
                assert len(value) == self._shape.numElements()
        else:
            raise NotImplementedError('Yet unsupported dtype: {}'
                                      .format(self._dtype))
        self._value = value
