import numpy as np

class Initializer:

    def __init__(self, *dims):
        self.dims = dims

    def initialize(self, init_method, **kwargs):
        """Initializes a numpy array according to a initialization method provided

        Args:
            str 'init_method': name of the initialization method to be used

        Returns:
            A numpy array with initial values according to init_method
        """

        if init_method == 'zero':
            return self.zero_initializer(self.dims)

        if init_method == 'constant':
            return self.constant_initializer (kwargs['constant'], self.dims)

        elif init_method == 'gaussian':
            return self.gaussian_initializer(self.dims, **kwargs)

        elif init_method == 'custom':
            return self.custom_initializer(kwargs['values'], self.dims)

        else:
            raise ValueError(f"Initialization method {init_method} is not a valid method.")

    @classmethod
    def zero_initializer(self, dims):
        """Initializes a numpy array with zeros

        Args:
            n-tuple 'dims': tuple of numpy array dimensions

        Returns:
            A numpy array with zeros
        """
        return np.zeros(dims)

    @classmethod
    def constant_initializer(self, constant, dims):
        """Initializes a numpy array with the same constant value

        Args:
            float 'constant': the constant value to be used in the initialization
            n-tuple 'dims': tuple of numpy array dimensions

        Returns:
            A numpy array with all the values equal to 'constant'
        """
        return np.ones(dims) * constant

    @classmethod
    def gaussian_initializer(self, dims, mean=0, std=1):
        """Initializes a numpy array according to the normal distribution

        Args:
            n-tuple 'dims': tuple of numpy array dimensions
            float 'mean': mean of the normal distribution
            float 'std': standard deviation of the normal distribution

        Returns:
            A numpy array with initial values sampled according to a normal distribution N(mean, std)
        """
        return np.random.randn(*dims)*std + mean

    @classmethod
    def custom_initializer(self, A, dims):
        assert np.equal(A.shape, dims).all()
        return A.copy()
