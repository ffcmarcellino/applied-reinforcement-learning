import numpy as np
from .initializers import Initializer

dims_list = ((1,), (5,), (3,4), (2,4,6))

def assert_shape(out, dims):
    assert out.shape == dims, f"Output shape ({out.shape}) does not match ({dims})."

def assert_values(out, values):
    assert (out == values).all()

def test_gaussian_initializer():
    initializer = Initializer(2, 3)
    out = initializer.initialize('gaussian')
    assert out.std() > 0
    assert out.mean() > initializer.initialize('gaussian', mean=-100, std=1).mean()

def test_initializers():

    for dims in dims_list:
        initializer = Initializer(*dims)
        assert_shape(initializer.initialize('zero'), dims)
        assert_shape(initializer.initialize('constant', constant = -3), dims)
        assert_shape(initializer.initialize('gaussian'), dims)

    initializer = Initializer(2, 3)
    assert_values(initializer.initialize('zero'), 0)
    assert_values(initializer.initialize('constant', constant = -3), -3)
    assert_values(initializer.initialize('custom', values=np.array([[1, 2, 3], [4, 5, 6]])), np.array([[1, 2, 3], [4, 5, 6]]))

def test_not_a_method():
    try:
        Initializer(3).initialize('not_a_method')
    except ValueError:
        pass
    else:
        raise Error("Method not_a_method should have raised a ValueError.")
