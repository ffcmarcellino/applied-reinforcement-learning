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

        elif init_method == 'uniform':
            return self.uniform_initializer(self.dims, **kwargs)

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
    def uniform_initializer(self, dims, low, high, one_hot=False):
        """Initializes a numpy array according to the uniform distribution in the range [low, high)

        Args:
            n-tuple 'dims': tuple of numpy array dimensions
            int 'low': lower bound of the distribution range, inclusive
            int 'high': upper bound of the distribution range, exclusive
            bool 'one_hot': whether the output is one-hot encoded or not

        Returns:
            A numpy array with initial values sampled according to a uniform distribution U(low, high) and one-hot encoded depending on the
            'one_hot' input argument
        """
        init_values = np.random.randint(low, high, dims)

        if one_hot:
            one_hot_cols = init_values.flatten()
            init_values = np.zeros((len(one_hot_cols), high-low)).astype(int)
            init_values[np.arange(len(one_hot_cols)), one_hot_cols] = 1
            return init_values.reshape((*dims, -1))

        return init_values

    @classmethod
    def custom_initializer(self, A, dims):
        """Initializes a numpy array with custom values given by array A

        Args:
            numpy.array 'A': array with the custom initial values
            n-tuple 'dims': tuple of numpy array dimensions

        Returns:
            A numpy array with initial values copied from A
        """
        assert np.equal(A.shape, dims).all()
        return A.copy()
