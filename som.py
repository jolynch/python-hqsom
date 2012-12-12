#Implementation of a Self Organizing Map
import numpy as np
import random

#Represents a single codebook of output_size units that are of size input_size
# EG: We want to classify 5d vectors and we want the SOM to output a 4d activation vector
#  => SOM(5,4)
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
        self.mse_ema = 1e-20
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

    #Adaptive Neighborhood_function: h_ib(t)
    # This basically tells us how close we are to the input and bmu
    # NOTE: Currently we use a gaussian, a mexican hat function might be better
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
        eff_zero = 1e-20
        if mse_bmu < eff_zero:
            return eff_zero
        if self.pure:
            val =  max(np.exp(- abs(unit_i - unit_bmu)**2 / (mse_bmu * spread)), eff_zero)
        else:
            t = abs(unit_i - unit_bmu)**2
            val = max((1-t/spread)*np.exp(- t / (mse_bmu * spread)), eff_zero)
        return val


    def update(self, unit_input, rate, spread):
        """
        SOM update rule

        TODO: select spread based on variance as per the paper

        Args:
            unit_input - An input vector of size N
            rate - (gamma) the learning rate
            spread - (sigma) the variance of the neighborhood function

        For eeach w_i in the map space, apply the following update rule
            w_i(t1) = w_i(t) + rate * h_ib(t)*(x(t)-w_i(t))
        """
        if len(unit_input) != self.units.shape[1]:
            print "Input of length {0} incompatible with SOM length {1}".format(len(unit_input),self.units.shape[1])
            return
        bmu_index = self.bmu(unit_input)
        bmu = self.units[bmu_index]
        mse = self.mse(unit_input)

        for weight_index in range(len(self.units)):
            w_t = self.units[weight_index]
            self.units[weight_index] = w_t + rate*self.nb_func(weight_index, bmu_index, mse, unit_input, spread)*(unit_input-w_t)

        if not self.pure:
            if mse / self.mse_ema > 10:
                self.t = 0
                print "Unexpectedly Large Miss!! Stealing BMU {}".format(bmu_index)
                print "Used effective sigma {} for update".format(.5)
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
        return (1.0/len(unit_input)) * np.linalg.norm(unit_input - bmu)**2

    def activation_vector(self, unit_input, continuous = False):
        val = np.zeros(len(self.units))
        if continuous:
            if self.pure:
                #Since we want the paper's implementation, use what we think their activation function is
                val = np.array([1.0/np.linalg.norm(unit_input-unit)**2 for unit in self.units])
                val /= np.linalg.norm(val)
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
            bmu_index = self.bmu(unit_input)
            val[bmu_index] = 1
        return val

class RSOM(SOM):
    """
    Recursive SOM, basically makes the update rule time dependent
    This should theoretically capture time dependencies

    Major difference to a SOM is that we have a first difference matrix as well
    as the standard units matrix
    """

    def __init__(self, input_size, output_size, pure=False):
        super(RSOM, self).__init__(input_size, output_size, pure)
        self.differences = np.zeros((output_size, input_size))

    def bmu_r(self, unit_input):
        distances = [np.linalg.norm(diff) for diff in self.differences]
        return np.argmin(distances)

    def reset(self):
        """
        So that we don't have to flush the RSOM with 0s all the damn time
        """
        self.differences = np.zeros(self.differences.shape)
        r_i = np.random.randint(len(self.differences))
        self.differences[r_i] = .001

    def update(self, unit_input, rate, spread, time_decay):
        """
        RSOM update rule
          Parameters same as SOM update rule except for time_decay
          time_decay : (alpha) the relative weight we give to past samples vs current samples

        y_i(t1) = (1-alpha)*y_i(t)+alpha*(x(t)-w_i(t))
        w_i(t1) = w_i(t) + rate * h_ibr(t)*y(t)
        """

        if len(unit_input) != self.units.shape[1]:
            print "Input of length {0} incompatible with SOM length {1}".format(len(unit_input),self.units.shape[1])
            return

        bmu_r_index = self.bmu_r(unit_input)
        bmu_r = self.units[bmu_r_index]
        mse = self.mse(unit_input)
        for weight_index in range(len(self.units)):
            w_t = self.units[weight_index]
            y_t = self.differences[weight_index]
            self.units[weight_index] = w_t + rate*self.nb_func(weight_index, bmu_r_index, mse, unit_input, spread)*y_t
            self.differences[weight_index] = (1-time_decay)*self.differences[weight_index] + time_decay * (unit_input-w_t)
        if not self.pure:
            rate = 2 * rate * np.exp(-np.power(self.time, 1/16.0))
            self.time += 1

