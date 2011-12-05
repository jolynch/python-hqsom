#Implementation of a Self Organizing Map
import numpy as np
import random

#Represents a single codebook of output_size units that are of size input_size
# EG: We want to classify 5d vectors and we want the SOM to output a 4d activation vector
#  => SOM(5,4)
class SOM(object):
    
    #TODO: Bottom of page 4, luttrell's method needed to expand size
    
    def __init__(self, input_size, output_size, pure=False):
        #random_array = [random.random() for i in range(input_size*output_size)]
        #random_array2 = [random.random() for i in range(input_size*output_size)]
        #random_array = np.array(random_array) + np.array(random_array2)[::-1]
        #random_array = random_array/2.0
        
        #self.units = np.array(random_array).reshape((output_size, input_size))
        self.units = np.random.random((output_size, input_size))
        self.mse_ema = 1e-20
        #If we consider this a non-pure SOM, use our "improvements"
        self.pure = pure

    #Best matching unit, returns the index of the bmu
    def bmu(self, unit_input):
        distances = [np.linalg.norm(unit - unit_input) for unit in self.units]
        return np.argmin(distances)

    #Adaptive Neighborhood_function: h_ib(t)
    # This basically tells us how close we are to the input and bmu
    # NOTE: Currently we use a gaussian, a mexican hat function might be better
    def nb_func(self, unit_i, unit_bmu, mse_bmu, unit_input, spread):
        #What do we want to consider 0?
        eff_zero = 1e-20
        #e^(-oo) = 0
        if mse_bmu < eff_zero:
            return eff_zero
        val =  max(np.exp(- abs(unit_i - unit_bmu)**3 / (mse_bmu * spread)), eff_zero)
        return val

    #SOM update rule
    # unit_input : An input vector of size N
    # rate : (gamma) the learning rate
    # spread : (sigma) the variance of the neighborhood function
    # 
    # w_i(t1) = w_i(t) + rate * h_ib(t)*(x(t)-w_i(t))

    #TODO: select spread based on variance as per the paper

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
       
        #print "MSE: {}".format(mse)
        #print "EMA: {}".format(self.mse_ema)
        if not self.pure:
            if mse / self.mse_ema > 10:
                print "Massive Miss!! Stealing BMU {}".format(bmu_index)
                print "Used effective sigma {} for update".format(mse * spread)
                self.units[bmu_index] = self.units[bmu_index] + .5*(unit_input - self.units[bmu_index])
            self.mse_ema = max(.9*(self.mse_ema) + .1*(mse), 1e-20)



    def mse(self, unit_input, som_unit=None):
        bmu = None
        if som_unit != None:
            bmu = som_unit
        else:
            bmu = self.units[self.bmu(unit_input)]
        return (1.0/len(unit_input)) * np.linalg.norm(unit_input - bmu)**2

    def activation_vector(self, unit_input, continuous = False):
        val = np.zeros(len(self.units))
        if continuous:
            if self.pure or True:
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
            
        else:
            bmu_index = self.bmu(unit_input)
            val[bmu_index] = 1
        return val

# Recursive SOM, basically makes the update rule time dependent
# Should theoretically capture time dependencies
class RSOM(SOM):
    
    #Major difference to a SOM is that we have a first difference matrix as well
    # as the standard units matrix
    def __init__(self, input_size, output_size, pure=False):
        super(RSOM, self).__init__(input_size, output_size, pure)
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
        
        if not self.pure:
            if mse / self.mse_ema > 10:
                print "Massive Miss!! Stealing BMU {}".format(bmu_r_index)
                print "Used effective sigma {} for update".format(mse * spread)
                self.units[bmu_r_index] = self.units[bmu_r_index] + .5*(unit_input - self.units[bmu_r_index])
            self.mse_ema = max(.9*(self.mse_ema) + .1*(mse), 1e-20)

