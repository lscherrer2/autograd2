from collections import deque
from typing import Optional, Type, Union, Any
import numpy as np

class Tensor:

    data: np.ndarray
    grad: np.ndarray
    grad_fn: Optional['GradFn']
    
    def __init__ (self, data: np.ndarray, grad_fn: Optional['GradFn'] = None):
        assert isinstance(data, np.ndarray), "data must be a numpy array"
        assert grad_fn is None or isinstance(grad_fn, GradFn), "grad_fn must be a GradFn or None"
        
        self.data = data
        self.grad_fn = grad_fn
        self.grad = np.zeros_like(data)

    def backward (self) -> None:
        self.grad = np.ones_like(self.data)
        backward_fns = cgraph_bfs(self)
        for grad_fn in backward_fns:
            grad_fn.backward()
        return None
    
    def shape (self) -> tuple[int, ...]:
        return self.data.shape
    
    def dtype (self) -> Type:
        return self.data.dtype.type
    
    def acceptable_num_types (self) -> tuple[Type, ...]:
        return (self.dtype(), int, float, np.number)

    def reshape (self, shape: tuple[int, ...]) -> 'Tensor':
        assert isinstance(shape, tuple), "shape must be a tuple"
        assert len(shape) > 0, "shape must have at least one dimension"

        return Reshape_fn(self, shape).result

    def _check_types (self, other: Union['Tensor', Any]) -> None:
        assert isinstance(other, (Tensor, *self.acceptable_num_types())), "other must be a Tensor or of the same datatype"
        if isinstance(other, Tensor):
            assert self.shape() == other.shape(), "self and other must have the same shape"
            
    def __add__ (self, other: Union['Tensor', Any]) -> 'Tensor':
        self._check_types(other)
        
        if isinstance(other, Tensor):
            return Add_fn(self, other).result
        elif isinstance(other, self.acceptable_num_types()):
            return Add_Scalar_fn(self, other).result
        else:
            raise NotImplementedError("Support for operation with these datatypes is not implemented yet")
        
    def __sub__ (self, other: Union['Tensor', Any]) -> 'Tensor':
        self._check_types(other)

        if isinstance(other, Tensor):
            return Sub_fn(self, other).result
        elif isinstance(other, self.acceptable_num_types()):
            return Sub_Scalar_fn(self, other).result
        else:
            raise NotImplementedError("Support for operation with these datatypes is not implemented yet")

    def __mul__ (self, other: Union['Tensor', Any]) -> 'Tensor':
        self._check_types(other)

        if isinstance(other, Tensor):
            return Mul_fn(self, other).result
        elif isinstance(other, self.acceptable_num_types()):
            return Mul_Scalar_fn(self, other).result
        else:
            raise NotImplementedError("Support for operation with these datatypes is not implemented yet")
    
    def __truediv__ (self, other: Union['Tensor', Any]) -> 'Tensor':
        self._check_types(other)

        if isinstance(other, Tensor):
            return Div_fn(self, other).result
        elif isinstance(other, self.acceptable_num_types()):
            return Div_Scalar_fn(self, other).result
        else:
            raise NotImplementedError("Support for operation with these datatypes is not implemented yet")

    def __pow__ (self, other: Union['Tensor', Any]) -> 'Tensor':
        self._check_types(other)

        if isinstance(other, Tensor):
            return Pow_fn(self, other).result
        elif isinstance(other, self.acceptable_num_types()):
            return Pow_Scalar_fn(self, other).result
        else:
            raise NotImplementedError("Support for operation with these datatypes is not implemented yet")

    def __matmul__ (self, other: 'Tensor') -> 'Tensor':
        assert isinstance(other, Tensor), "other must be a Tensor"
        assert len(self.shape()) == 2, "self must be a 2D Tensor"
        assert len(other.shape()) == 2, "other must be a 2D Tensor"
        assert self.shape()[1] == other.shape()[0], "Tensors must have compatible shapes for matrix multiplication"

    def __radd__ (self, other: Union['Tensor', Any]) -> 'Tensor':
        return self.__add__(other)

    def __rsub__ (self, other: Union['Tensor', Any]) -> 'Tensor':
        return self.__mul__(self.__sub__(other), self.dtype()(-1))

    def __rmul__ (self, other: Union['Tensor', Any]) -> 'Tensor':
        return self.__mul__(other)

    def __rtruediv__ (self, other: Union['Tensor', Any]) -> 'Tensor':
        return self.__pow__(self.__truediv__(other), self.dtype()(-1))
    
    def __rpow__ (self, other: Union['Tensor', Any]) -> 'Tensor':
        raise NotImplementedError("Support for operation implemented yet. Consider reversing the operation.")

    def __iadd__ (self, other: Union['Tensor', Any]) -> 'Tensor':
        self._check_types(other)
        
        if isinstance(other, Tensor):
            self = Add_fn(self, other).result
            return self
        elif isinstance(other, self.acceptable_num_types()):
            self = Add_Scalar_fn(self, other).result
            return self
        else:
            raise NotImplementedError("Support for operation with these datatypes is not implemented yet")
       
    def __isub__ (self, other: Union['Tensor', Any]) -> 'Tensor':
        self._check_types(other)
        
        if isinstance(other, Tensor):
            self = Sub_fn(self, other).result
            return self
        elif isinstance(other, self.acceptable_num_types()):
            self = Sub_Scalar_fn(self, other).result
            return self
        else:
            raise NotImplementedError("Support for operation with these datatypes is not implemented yet")
       
    def __imul__ (self, other: Union['Tensor', Any]) -> 'Tensor':
        self._check_types(other)
        
        if isinstance(other, Tensor):
            self = Mul_fn(self, other).result
            return self
        elif isinstance(other, self.acceptable_num_types()):
            self = Mul_Scalar_fn(self, other).result
            return self
        else:
            raise NotImplementedError("Support for operation with these datatypes is not implemented yet")
       
    def __itruediv__ (self, other: Union['Tensor', Any]) -> 'Tensor':
        self._check_types(other)
        
        if isinstance(other, Tensor):
            self = Div_fn(self, other).result
            return self
        elif isinstance(other, self.acceptable_num_types()):
            self = Div_Scalar_fn(self, other).result
            return self
        else:
            raise NotImplementedError("Support for operation with these datatypes is not implemented yet")

def tensor (*args, **kwargs) -> Tensor:
    return Tensor(np.array(*args, **kwargs))

def astensor (*args, **kwargs) -> Tensor:
    return Tensor(np.asarray(*args, **kwargs))

def from_numpy (data: np.ndarray) -> Tensor:
    assert isinstance(data, np.ndarray), "data must be a numpy array"
    return Tensor(data)

def to_numpy (tensor: Tensor) -> np.ndarray:
    assert isinstance(tensor, Tensor), "tensor must be a Tensor"
    return tensor.data

def cgraph_bfs (tensor: Tensor) -> list["GradFn"]:

    seen_grad_fns: set[GradFn] = set()
    grad_fns: deque[GradFn] = deque()

    active_tensors: deque[Tensor] = deque([tensor])

    while active_tensors:

        next_grad_fns: deque[GradFn] = deque()

        for tensor in active_tensors:

            # chk grad_fn for cyclic graph
            if tensor.grad_fn in seen_grad_fns:
                continue

            # chk if dead end
            if tensor.grad_fn is None:
                continue

            # add grad_fn to the result and to the next bfs layer
            seen_grad_fns.add(tensor.grad_fn)
            grad_fns.append(tensor.grad_fn)
            next_grad_fns.append(tensor.grad_fn)
        
        active_tensors = deque()

        for grad_fn in next_grad_fns:
            active_tensors.extend(grad_fn.creators)
    
    return grad_fns

class GradFn:

    creators: list[Tensor]
    result: Tensor

    def __init__ (self, *args: Tensor) -> None:
        for element in args:
            assert isinstance(element, Tensor), "creators must be a list of Tensors"
        
        self.creators = list(args)

    def backward (self) -> None:
        assert isinstance(self.result.grad, np.ndarray), "result must have a gradient"

    def _check_numeric_types (self, other: Any) -> None:
        assert isinstance(other, (int, float, np.number)), "other must be a number"

class MSE_Loss_fn (GradFn):

    num_elements: int
    y_pred: Tensor
    y_true: Tensor

    def __init__ (self, y_pred: Tensor, y_true: Tensor) -> None:
        assert y_pred.shape() == y_true.shape(), "y_pred and y_true must have the same shape"
        assert isinstance(y_pred, Tensor), "y_pred must be a Tensor"
        assert isinstance(y_true, Tensor), "y_true must be a Tensor"

        super().__init__(y_pred, y_true)
        self.num_elements = np.prod(y_pred.shape())
        self.y_pred = y_pred
        self.y_true = y_true
        self.result = Tensor(np.array(np.mean(y_pred.data - y_true.data)**2), grad_fn=self)

    def backward (self) -> None:
        super().backward()
        assert np.prod(self.result.shape()) == 1, "result must have no more than one element"

        self.y_pred.grad += self.result.grad.reshape(-1)[0] * 2 * (self.y_pred.data - self.y_true.data) / self.num_elements
        self.y_true.grad += self.result.grad.reshape(-1)[0] * 2 * (self.y_true.data - self.y_pred.data) / self.num_elements

class Add_fn (GradFn):

    def __init__ (self, a: Tensor, b: Tensor) -> None:
        assert isinstance(a, Tensor), "a must be a Tensor"
        assert isinstance(b, Tensor), "b must be a Tensor"
        assert a.shape() == b.shape(), "a and b must have the same shape"

        super().__init__(a, b)
        self.a = a
        self.b = b
        self.result = Tensor(a.data + b.data, grad_fn=self)
    
    def backward (self) -> None:
        super().backward()

        self.a.grad += self.result.grad
        self.b.grad += self.result.grad
    
class Sub_fn (GradFn):

    def __init__ (self, a: Tensor, b: Tensor) -> None:
        assert isinstance(a, Tensor), "a must be a Tensor"
        assert isinstance(b, Tensor), "b must be a Tensor"
        assert a.shape() == b.shape(), "a and b must have the same shape"

        super().__init__(a, b)
        self.a = a
        self.b = b
        self.result = Tensor(a.data - b.data, grad_fn=self)
    
    def backward (self) -> None:
        super().backward()

        self.a.grad += self.result.grad
        self.b.grad -= self.result.grad

class Mul_fn (GradFn):

    def __init__ (self, a: Tensor, b: Tensor) -> None:
        assert isinstance(a, Tensor), "a must be a Tensor"
        assert isinstance(b, Tensor), "b must be a Tensor"
        assert a.shape() == b.shape(), "a and b must have the same shape"

        super().__init__(a, b)
        self.a = a
        self.b = b
        self.result = Tensor(a.data * b.data, grad_fn=self)
    
    def backward (self) -> None:
        super().backward()

        self.a.grad += self.result.grad * self.b.data
        self.b.grad += self.result.grad * self.a.data

class Div_fn (GradFn):

    def __init__ (self, a: Tensor, b: Tensor) -> None:
        assert isinstance(a, Tensor), "a must be a Tensor"
        assert isinstance(b, Tensor), "b must be a Tensor"
        assert a.shape() == b.shape(), "a and b must have the same shape"

        super().__init__(a, b)
        self.a = a
        self.b = b
        self.result = Tensor(a.data / b.data, grad_fn=self)
    
    def backward (self) -> None:
        super().backward()

        self.a.grad += self.result.grad / self.b.data
        self.b.grad -= self.result.grad * self.a.data / (self.b.data**2)

class Add_Scalar_fn (GradFn):

    def __init__ (self, a: Tensor, b: Any) -> None:
        assert isinstance(a, Tensor), "a must be a Tensor"
        self._check_numeric_types(b)

        super().__init__(a)
        self.a = a
        self.b = b
        self.result = Tensor(a.data + b, grad_fn=self)
    
    def backward (self) -> None:
        super().backward()

        self.a.grad += self.result.grad

class Sub_Scalar_fn (GradFn):

    def __init__ (self, a: Tensor, b: Any) -> None:
        assert isinstance(a, Tensor), "a must be a Tensor"
        self._check_numeric_types(b)

        super().__init__(a)
        self.a = a
        self.b = b
        self.result = Tensor(a.data - b, grad_fn=self)
    
    def backward (self) -> None:
        super().backward()

        self.a.grad += self.result.grad

class Mul_Scalar_fn (GradFn):
    
        def __init__ (self, a: Tensor, b: Any) -> None:
            assert isinstance(a, Tensor), "a must be a Tensor"
            self._check_numeric_types(b)
    
            super().__init__(a)
            self.a = a
            self.b = b
            self.result = Tensor(a.data * b, grad_fn=self)
        
        def backward (self) -> None:
            super().backward()
    
            self.a.grad += self.result.grad * self.b

class Div_Scalar_fn (GradFn):

    def __init__ (self, a: Tensor, b: Any) -> None:
        assert isinstance(a, Tensor), "a must be a Tensor"
        self._check_numeric_types(b)
        assert b != 0, "b cannot be zero"

        super().__init__(a)
        self.a = a
        self.b = b
        self.result = Tensor(a.data / b, grad_fn=self)
    
    def backward (self) -> None:
        super().backward()

        self.a.grad += self.result.grad / self.b

class Pow_fn (GradFn):

    def __init__ (self, a: Tensor, b: Tensor) -> None:
        assert isinstance(a, Tensor), "a must be a Tensor"
        assert isinstance(b, Tensor), "b must be a Tensor"

        super().__init__(a, b)
        self.a = a
        self.b = b
        self.result = Tensor(a.data ** b.data, grad_fn=self)
    
    def backward (self) -> None:
        super().backward()

        self.a.grad += self.result.grad * self.b.data * (self.a.data ** (self.b.data - 1))
        self.b.grad += self.result.grad * np.log(self.a.data) * self.result.data

class Pow_Scalar_fn (GradFn):

    def __init__ (self, a: Tensor, b) -> None:
        assert isinstance(a, Tensor), "a must be a Tensor"
        self._check_numeric_types(b)

        super().__init__(a)
        self.a = a
        self.b = b
        self.result = Tensor(a.data ** b, grad_fn=self)
    
    def backward (self) -> None:
        super().backward()

        self.a.grad += self.result.grad * self.b * (self.a.data ** (self.b - 1))

class Matmul_fn (GradFn):
    
        def __init__ (self, a: Tensor, b: Tensor) -> None:
            assert isinstance(a, Tensor), "a must be a Tensor"
            assert isinstance(b, Tensor), "b must be a Tensor"
            assert len(a.shape()) == 2, "a must be a 2D Tensor"
            assert len(b.shape()) == 2, "b must be a 2D Tensor"
            assert a.shape()[1] == b.shape()[0], "Tensors must have compatible shapes for matrix multiplication"
    
            super().__init__(a, b)
            self.a = a
            self.b = b
            self.result = Tensor(np.matmul(a.data, b.data), grad_fn=self)
        
        def backward (self) -> None:
            super().backward()
    
            self.a.grad += np.matmul(self.result.grad, self.b.data.T)
            self.b.grad += np.matmul(self.a.data.T, self.result.grad)

class Reshape_fn (GradFn):

    def __init__ (self, a: Tensor, shape: tuple[int, ...]) -> None:
        assert isinstance(a, Tensor), "a must be a Tensor"
        assert isinstance(shape, tuple), "shape must be a tuple"
        assert len(shape) > 0, "shape must have at least one dimension"

        self.old_shape = a.shape()
        self.new_shape = shape

        assert np.prod(self.old_shape) == np.prod(self.new_shape), "a and shape must have the same number of elements"

        super().__init__(a)
        self.a = a

        self.shape = shape
        self.result = Tensor(a.data.reshape(self.new_shape), grad_fn=self)
    
    def backward (self) -> None:
        super().backward()
        self.a.grad = self.result.grad.reshape(self.old_shape)
    
    @classmethod
    def __call__ (cls, a: Tensor, shape: tuple[int, ...]) -> Tensor:
        return cls(a, shape)
