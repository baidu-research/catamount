from enum import Enum
import TensorShape

class DataType(Enum):
    bool = 0
    int8 = 1
    int16 = 2
    int32 = 3
    int64 = 4
    float16 = 5
    float32 = 6
    float64 = 7


class Tensor:
    def __init__(self, shape, dtype=DataType.float32):
        self._shape = shape
        self._dtype = dtype

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

