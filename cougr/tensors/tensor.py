from enum import Enum
from .tensor_shape import TensorShape

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
        if self._shape.rank == 0:
            assert isinstance(value, int) or isinstance(value, float)
            self._value = value
        else:
            raise NotImplementedError('Tensor setValue to 1+ rank value')
