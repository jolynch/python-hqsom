#Implementation of a Self Organizing Map
import numpy as np


#Represents a single codebook of output_size units that are of size input_size
# EG: We want to classify 5d vectors and we want the SOM to output a 4d activation vector
#  => SOM(5,4)
class SOM:
    def __init__(self, input_size, output_size):
        self.units = np.random.random((output_size, input_size))

    #Best matching unit, returns the index of the bmu
    def bmu(self, unit_input):
        distances = [np.linalg.norm(unit - unit_input) for unit in self.units]
        return np.argmin(distances)

    #Adaptive Neighborhood_function: h_ib(t)
    # This basically tells us how close we are to the input and bmu
    def nb_func(self, unit_i, unit_bmu, mse_bmu, unit_input, spread):
        return np.exp(- abs(unit_i - unit_bmu)**2 / (mse_bmu * spread))

    #SOM update rule
    # unit_input : An input vector of size N
    # rate : (gamma) the learning rate, valid values from 0 to 1
    # spread : (sigma) the variance of the neighborhood function, valid values from 0 to 1 
    # 

    def update(self, unit_input, rate, spread):
        if len(unit_input) != self.units.shape[1]:
            print "Input of length {0} incompatible with SOM length {1}".format(len(unit_input),self.units.shape[1])
            return

        print "Updating on input: {}".format( unit_input)
        bmu_index = self.bmu(unit_input)
        bmu = self.units[bmu_index]
        mse = (1.0/len(unit_input)) * np.linalg.norm(unit_input - bmu)**2
        print "MSE before update: {}".format(mse)
        for weight_index in range(len(self.units)):
            w_t = self.units[weight_index]
            self.units[weight_index] = w_t + rate*self.nb_func(weight_index, bmu_index, mse, unit_input, spread)*(unit_input-w_t)

class RSOM(SOM):
    def __init__(self, input_size, output_size):
        super(RSOM, self).__init__(input_size, output_size)
        self.differences = np.zeros(output_size)

if __name__ == "__main__":
    som = SOM(3,3)
    rate,spread = .7, .2
    test_data= [(.2,.2,.2),(0,0,.2),(.8,.8,.1),(.9,.9,.9)]
    test_data = np.array([np.array(i) for i in test_data])
    som.update(np.array([1,1,1,1]),rate,spread)
    print som.units
    for i in range(10):
        for t in test_data:
            som.update(t,rate,spread)
    print som.units

