import numpy as np
from som import SOM

class RSOM(SOM):
    """
    Recursive SOM, basically makes the update rule time dependent
    This should theoretically capture time dependencies

    Major difference to a SOM is that we have a first difference matrix as well
    as the standard units matrix
    """

    def __init__(self, input_size, output_size, pure=False):
        """
        Initialize the data structures required for the RSOM of input input_size
        and output of output_size

        Members:
            differences - First difference matrix, stores the recursive
                          differences for each update
        """
        super(RSOM, self).__init__(input_size, output_size, pure)
        self.differences = np.zeros((output_size, input_size))

    def bmu_r(self, unit_input):
        """
        Calculate and return the recursive best matching unit, which is the concept vector
        closest to the unit input vector taking into account previous concept vectors

        Args:
            unit_input - The input data to examine

        Returns the *index* of the best matching unit in the mapspace
        """
        return np.argmin([np.linalg.norm(diff) for diff in self.differences])

    def reset(self):
        """
        Resets the recursive difference matrix.

        So that we don't have to flush the RSOM with 0s all the damn time
        """
        self.differences = np.zeros(self.differences.shape)
        r_i = np.random.randint(len(self.differences))
        self.differences[r_i] = .001 # This will automatically get selected as the BMU next time

    def update(self, unit_input, rate, spread, time_decay):
        """
        RSOM update rule

        y_i(t1) = (1-alpha) * y_i(t) + alpha * (x(t) - w_i(t))
        w_i(t1) = w_i(t) + rate * h_ibr(t) * y(t)

        Args: same as SOM update rule except for time_decay
          time_decay - (alpha) the relative weight we give to past samples vs current samples
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
            self.units[weight_index] = w_t + rate * self.nb_func(weight_index, bmu_r_index, mse, unit_input, spread) * y_t
            self.differences[weight_index] = (1-time_decay)*self.differences[weight_index] + time_decay * (unit_input - w_t)

        if not self.pure:
            # Make the RSOMs less plastic over time
            rate = 2 * rate * np.exp(-np.power(self.time, 1/16.0))
            self.time += 1

