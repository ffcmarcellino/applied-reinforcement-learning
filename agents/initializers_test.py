import numpy as np
from .initializers import Initializer

dims_list = ((1,), (5,), (3,4), (2,4,6))

def assert_custom_initializer(out, c, dims):
    assert out.shape == dims, f"Output shape ({out.shape}) does not match ({dims})."
    assert (out == c).all()

def assert_gaussian_initializer(initializer, dims):
    out = initializer.initialize('gaussian')
    assert out.shape == dims, f"Output shape ({out.shape}) does not match ({dims})."
    if len(dims) > 1:
        assert out.std() > 0
    assert out.mean() > initializer.initialize('gaussian', mean=-100, std=1).mean()

def test_initializers():

    for dims in dims_list:
        initializer = Initializer(*dims)
        assert_custom_initializer(initializer.initialize('zero'), 0, dims)
        assert_custom_initializer(initializer.initialize('constant', constant = -3), -3, dims)
        assert_gaussian_initializer(initializer, dims)

    initializer = Initializer(2, 3)
    assert_custom_initializer(initializer.initialize('custom', values=np.array([[1, 2, 3], [4, 5, 6]])), np.array([[1, 2, 3], [4, 5, 6]]), (2,3))

def test_not_a_method():
    try:
        Initializer(3).initialize('not_a_method')
    except ValueError:
        pass
    else:
        raise Error("Method not_a_method should have raised a ValueError.")

def test_gaussian_initializer():
    initializer = Initializer(3,4)
