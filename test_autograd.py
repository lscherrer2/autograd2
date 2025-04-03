import numpy as np
from autograd import Tensor, tensor, from_numpy, to_numpy
import unittest

class TestTensorOperations(unittest.TestCase):
    
    def test_basic_operations(self):
        """Test basic tensor operations like addition, subtraction, multiplication, division."""
        a = tensor([1.0, 2.0, 3.0])
        b = tensor([4.0, 5.0, 6.0])
        
        # Test addition
        c = a + b
        np.testing.assert_array_equal(c.data, np.array([5.0, 7.0, 9.0]))
        
        # Test subtraction
        c = a - b
        np.testing.assert_array_equal(c.data, np.array([-3.0, -3.0, -3.0]))
        
        # Test multiplication
        c = a * b
        np.testing.assert_array_equal(c.data, np.array([4.0, 10.0, 18.0]))
        
        # Test division
        c = a / b
        np.testing.assert_array_almost_equal(c.data, np.array([0.25, 0.4, 0.5]))
    
    def test_scalar_operations(self):
        """Test operations between tensors and scalars."""
        a = tensor([1.0, 2.0, 3.0])
        scalar = 2.0
        
        # Test scalar addition
        c = a + scalar
        np.testing.assert_array_equal(c.data, np.array([3.0, 4.0, 5.0]))
        
        # Test scalar subtraction
        c = a - scalar
        np.testing.assert_array_equal(c.data, np.array([-1.0, 0.0, 1.0]))
        
        # Test scalar multiplication
        c = a * scalar
        np.testing.assert_array_equal(c.data, np.array([2.0, 4.0, 6.0]))
        
        # Test scalar division
        c = a / scalar
        np.testing.assert_array_equal(c.data, np.array([0.5, 1.0, 1.5]))
    
    def test_power_operations(self):
        """Test power operations on tensors."""
        a = tensor([1.0, 2.0, 3.0])
        b = tensor([2.0, 2.0, 2.0])
        
        # Test power with tensor
        c = a ** b
        np.testing.assert_array_equal(c.data, np.array([1.0, 4.0, 9.0]))
        
        # Test power with scalar
        c = a ** 2
        np.testing.assert_array_equal(c.data, np.array([1.0, 4.0, 9.0]))
    
    def test_inplace_operations(self):
        """Test in-place operations."""
        a = tensor([1.0, 2.0, 3.0])
        b = tensor([4.0, 5.0, 6.0])
        
        # Test in-place addition
        a_copy = tensor([1.0, 2.0, 3.0])
        a_copy += b
        np.testing.assert_array_equal(a_copy.data, np.array([5.0, 7.0, 9.0]))
        
        # Test in-place subtraction
        a_copy = tensor([1.0, 2.0, 3.0])
        a_copy -= b
        np.testing.assert_array_equal(a_copy.data, np.array([-3.0, -3.0, -3.0]))
        
        # Test in-place multiplication
        a_copy = tensor([1.0, 2.0, 3.0])
        a_copy *= b
        np.testing.assert_array_equal(a_copy.data, np.array([4.0, 10.0, 18.0]))
        
        # Test in-place division
        a_copy = tensor([1.0, 2.0, 3.0])
        a_copy /= b
        np.testing.assert_array_almost_equal(a_copy.data, np.array([0.25, 0.4, 0.5]))


class TestAutograd(unittest.TestCase):
    
    def test_addition_backward(self):
        """Test gradient computation for addition operation."""
        a = tensor([1.0, 2.0, 3.0])
        b = tensor([4.0, 5.0, 6.0])
        
        c = a + b
        c.backward()
        
        # Gradient of c with respect to a should be 1.0
        np.testing.assert_array_equal(a.grad, np.array([1.0, 1.0, 1.0]))
        # Gradient of c with respect to b should be 1.0
        np.testing.assert_array_equal(b.grad, np.array([1.0, 1.0, 1.0]))
    
    def test_multiplication_backward(self):
        """Test gradient computation for multiplication operation."""
        a = tensor([1.0, 2.0, 3.0])
        b = tensor([4.0, 5.0, 6.0])
        
        c = a * b
        c.backward()
        
        # Gradient of c with respect to a should be b
        np.testing.assert_array_equal(a.grad, np.array([4.0, 5.0, 6.0]))
        # Gradient of c with respect to b should be a
        np.testing.assert_array_equal(b.grad, np.array([1.0, 2.0, 3.0]))
    
    def test_computational_graph(self):
        """Test more complex computational graph."""
        a = tensor([2.0])
        b = tensor([3.0])
        
        # c = a * b
        c = a * b
        
        # d = c + a = a * b + a = a * (b + 1)
        d = c + a
        
        d.backward()
        
        # Gradient of d with respect to a should be (b + 1) = 4.0
        np.testing.assert_array_equal(a.grad, np.array([4.0]))
        # Gradient of d with respect to b should be a = 2.0
        np.testing.assert_array_equal(b.grad, np.array([2.0]))


if __name__ == "__main__":
    unittest.main()