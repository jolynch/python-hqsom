#Implementation of a Self Organizing Map
import numpy as np

#Represents a single codebook of output_size units that are of size input_size
# EG: We want to classify 5d vectors and we want the SOM to output a 4d activation vector
#  => SOM(5,4)
class SOM(object):
    def __init__(self, input_size, output_size):
        self.units = np.random.random((output_size, input_size))

    #Best matching unit, returns the index of the bmu
    def bmu(self, unit_input):
        distances = [np.linalg.norm(unit - unit_input) for unit in self.units]
        return np.argmin(distances)

    #Adaptive Neighborhood_function: h_ib(t)
    # This basically tells us how close we are to the input and bmu
    def nb_func(self, unit_i, unit_bmu, mse_bmu, unit_input, spread):
        #What do we want to consider 0?
        eff_zero = 1e-12
        #e^(-oo) = 0
        if mse_bmu < eff_zero:
            return 0
        val =  np.exp(- abs(unit_i - unit_bmu)**2 / (mse_bmu * spread))
        #
        if val < eff_zero:
            return 0
        return val

    #SOM update rule
    # unit_input : An input vector of size N
    # rate : (gamma) the learning rate, valid values from 0 to 1
    # spread : (sigma) the variance of the neighborhood function, valid values from 0 to 1 
    # 
    # w_i(t1) = w_i(t) + rate * h_ib(t)*(x(t)-w_i(t))

    def update(self, unit_input, rate, spread):
        if len(unit_input) != self.units.shape[1]:
            print "Input of length {0} incompatible with SOM length {1}".format(len(unit_input),self.units.shape[1])
            return
        bmu_index = self.bmu(unit_input)
        bmu = self.units[bmu_index]
        mse = self.mse(unit_input)
        for weight_index in range(len(self.units)):
            w_t = self.units[weight_index]
            self.units[weight_index] = w_t + rate*self.nb_func(weight_index, bmu_index, mse, unit_input, spread)*(unit_input-w_t)

    def mse(self, unit_input):
        bmu = self.units[self.bmu(unit_input)]
        return (1.0/len(unit_input)) * np.linalg.norm(unit_input - bmu)**2


    def activation_vector(self, unit_input, continuous = False):
        val = np.zeros(len(self.units))
        if continuous:
            val = [np.linalg.norm(unit-unit_input) for unit in self.units]
            min_val = min(val)
            val = np.array([ min_val / i for i in val])
        else:
            bmu_index = self.bmu(unit_input)
            val[bmu_index] = 1
        return val

# Recursive SOM, basically makes the update rule time dependent
# Should theoretically capture time dependencies
class RSOM(SOM):
    
    #Major difference to a SOM is that we have a first difference matrix as well
    # as the standard units matrix
    def __init__(self, input_size, output_size):
        super(RSOM, self).__init__(input_size, output_size)
        self.differences = np.zeros((output_size, input_size))

    def bmu_r(self, unit_input):
        distances = [np.linalg.norm(diff) for diff in self.differences]
        return np.argmin(distances)
    
    #RSOM update rule
    # Parameters same as SOM update rule except for time_decay
    # time_decay : (alpha) the relative weight we give to past samples vs current samples
    # 
    #  y_i(t1) = (1-alpha)*y_i(t)+alpha*(x(t)-w_i(t))
    #  w_i(t1) = w_i(t) + rate * h_ibr(t)*y(t)

    def update(self, unit_input, rate, spread, time_decay):
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

# Some test as I go, need to do more serious unit testing
# TODO: Write actual tests
if __name__ == "__main__":
    som = RSOM(3,10)
    rate,spread,alpha = 1, 1, 1
    test_data= [(.2,.2,.2),(0,0,.2),(.8,.8,.1),(.9,.9,.9)]
    test_data = np.array([np.array(i) for i in test_data])
    som.update(np.array([1,1,1,1]),rate,spread,alpha)
    print som.units
    index = 0
    for j in range(400):
        rate *= .8
        
        if np.random.random() > .7:
            index = np.random.randint(0,4)

        som.update(test_data[index],rate,spread,alpha)
    print som.units
    for t in test_data:
        print t
        print som.activation_vector(t )
        print som.activation_vector(t, True)
