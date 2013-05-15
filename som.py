#Implementation of a Self Organizing Map
import numpy as np
import random

EFFECTIVE_ZERO = 1e-20

class SOM(object):
    """
    Self Organizing Map

    Implementation of a spatial clustering algorithm whereby there are
    inputs of degree input_size that are spatially clustered into the
    output space of degree output_size.  This means that vectors that are
    close to each other in input space should result in near identical
    output vectors.

    TODO:
        - Bottom of page 4, luttrell's method needed to expand size
          of map size depending on usage
    """

    def __init__(self, input_size, output_size, pure=False):
        """
        Initialize the data structures required for the SOM of input input_size
        and output of output_size

        Members:
            units - A numpy matrix of dimension input_size x output_size
            mse_ema - An exponential moving average of the MSE between the
                      BMU and the input data.
            pure - A flag to indicate whether the paper's implementation
                   should be used or if our improvements should be used
            time - How many timesteps have passed
        """
        self.units = np.random.random((output_size, input_size))
        self.mse_ema = EFFECTIVE_ZERO
        self.pure = pure
        self.time = 0

    def bmu(self, unit_input):
        """
        Calculate and return the best matching unit, which is the concept vector
        closest to the unit input vector

        Args:
            unit_input - The input data to examine

        Returns the *index* of the best matching unit in the mapspace
        """

        distances = [np.linalg.norm(unit - unit_input) for unit in self.units]
        return np.argmin(distances)

    def nb_func(self, unit_i, unit_bmu, mse_bmu, unit_input, spread):
        """
        Calculate the adaptive neighborhood function h_ib(t), which is a measure
        of how close the provided map unit is to the provided input, relative to
        the best matching unit.

        The basic idea is h_ib(t) = exp(-(unit_i, unit_bmu) /
                                        (mse(unit_input, unit_bmu)*spread))

        although in our improved version we use a mexican hat function
        instead of a gaussian

        Args:
            unit_i - The map unit *index* to compare to the input and the bmu
            unit_bmu - The map unit *index* of the bmu
            mse_bmu - The MSE of the bmu and the input (computed once and passed in)
            unit_input - The input unit
            spread - The spread parameter that decided how wide the tails are

        Returns val: The value of the neighborhood function

        """
        # For numerical stability we need to pick something that is effectively zero
        # and not actually do the computation for various very wonky values
        if mse_bmu < EFFECTIVE_ZERO:
            return EFFECTIVE_ZERO

        t_squared = abs(unit_i - unit_bmu)**2
        if self.pure:
            # ~Gaussian
            val =  max(np.exp(-t_squared / (mse_bmu * spread)), EFFECTIVE_ZERO)
        else:
            # ~Mexican Hat
            val = max((1 - t_squared / spread) * np.exp(-t_squared / (mse_bmu * spread)), EFFECTIVE_ZERO)
        return val


    def update(self, unit_input, rate, spread):
        """
        SOM update rule

        TODO: select spread based on variance as per the paper

        Args:
            unit_input - An input vector of size N
            rate - (gamma) the learning rate
            spread - (sigma) the variance of the neighborhood function

        For each w_i in the map space, apply the following update rule
            w_i(t1) = w_i(t) + rate * h_ib(t)*(x(t)-w_i(t))
        """
        if len(unit_input) != self.units.shape[1]:
            print "Input of length {0} incompatible with SOM length {1}".format(len(unit_input),self.units.shape[1])
            return
        bmu_index = self.bmu(unit_input)
        bmu = self.units[bmu_index]
        mse = self.mse(unit_input, bmu)

        for weight_index in range(len(self.units)):
            w_t = self.units[weight_index]
            self.units[weight_index] = w_t + rate*self.nb_func(weight_index, bmu_index, mse, unit_input, spread)*(unit_input-w_t)

        if not self.pure:
            # When we have particularly unexpected MSE compared to what we are used to, we move a BMU
            # almost entirely towards the input, essentially forcing the input into the concept space
            # Caution: This can cause overfitting if used incorrectly
            if mse / self.mse_ema > 10:
                self.t = 0
                #print "Unexpectedly Large Miss!! Stealing BMU {}".format(bmu_index)
                #print "Used effective sigma {} for update".format(.5)
                self.units[bmu_index] = self.units[bmu_index] + .5*(unit_input - self.units[bmu_index])
            self.mse_ema = max(.9*(self.mse_ema) + .1*(mse), 1e-20)

    def mse(self, unit_input, som_unit=None):
        """
        Calculate the Mean Squared Error between unit_input
        and som_unit
        """
        bmu = None
        if som_unit != None:
            bmu = som_unit
        else:
            bmu = self.units[self.bmu(unit_input)]
        return (1.0 /len(unit_input)) * np.linalg.norm(unit_input - bmu)**2

    def activation_vector(self, unit_input, continuous = False):
        """
        SOM activation vector for a given input vector

        Args:
            unit_input - The input vector to test activation for
            continuous - If True, return a continuous activation vector based on MSE,
                         if False, return a discrete activation vector with the BMU = 1
                         and everything else = 0

        """
        val = np.zeros(len(self.units))
        if continuous:
            if self.pure:
                # Since we want the paper's implementation, use what we think their activation function is
                val = np.array([1.0 / np.linalg.norm(unit_input-unit)**2 for unit in self.units])
            else:
                # Since we minimize mse, I use that as the normalization for continuous
                # activation vectors.  The square is to decrease the values of non matching
                # units more
                mse_bmu = self.mse(unit_input)**3
                for i in range(len(self.units)):
                    val[i] = self.mse(unit_input, self.units[i])**3
                val = np.array([ mse_bmu / v for v in val])
            val /= np.linalg.norm(val)
        else:
            val[self.bmu(unit_input)] = 1
        return val

