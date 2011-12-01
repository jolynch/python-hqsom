'''
Module for making HQSOM base units out of RSOMs and SOMs,
and wiring together the HQSOM base units into hierarchies.
'''

from som import *



'''
HQSOM base unit, comprised of a SOM feeding into an RSOM
'''
class HQSOM(object):
    
    '''
    @param som_input_size - size of input, how much data do we have
    @param som_map_size - # of spatial entities you want to be able to identify
    @param rsom_map_size - # of spatial-temporal entities you want to be able to identify
    @param initMapUnits - specify if fewer than outSize are desired
    @param grow - grow internal map using Luttrell's method; requires support
                    by underlying SOM/RSOM implementation
   
    '''
    #NOTE: intially we only support the RSOM outputing a single number: it's bmu index
    def __init__(self, som_input_size, som_map_size, rsom_map_size,
                 initMapUnits=None, grow=False):
        self.som = SOM(som_input_size, som_map_size)
        self.rsom = RSOM(som_map_size, rsom_map_size)

    '''
    @param unit_input - the test data
    @param gamma_som - the learning rate of the som
    @param gamma_rsom - the learning rate of the rsom
    @param sigma - the spread of the neighborhood function
    @param alpha - the relative time value of data
    '''

    def update(self, unit_input, gamma_som=.3, gamma_rsom=.3, sigma_som=.8, sigma_rsom=.8, alpha=.5):
        #print "Training on {}".format(unit_input)
        self.som.update(unit_input, gamma_som, sigma_som)
        som_output = self.som.activation_vector(unit_input, True)
        self.rsom.update(som_output, gamma_rsom, sigma_rsom, alpha)

    def activation_vector(self, unit_input, continuous=False):
        som_output = self.som.activation_vector(unit_input, True)
        if continuous:
            return self.rsom.activation_vector(som_output, True)
        else:
            return self.rsom.bmu(som_output)




'''
Configuration object to create hierarchies of base units
'''
class HierarchyConfig(object):
    def __init__(self):
        pass


'''
Assumes the input field is square, that the 

'''
class Simple2dHierarchyConfig(object):
    def __init__(self):
        pass





'''
Base class for hierarchies of HQSOM units
'''
class Hierarchy(object):
    def __init__(self):
        pass
    def update(self, input):
        pass
    def bmu(self):
        pass


'''
Hierarchies in which internal and upper layers operate only on
best-matching unit (BMU) *indices*, rather than map unit activation vectors.
The upper units operate on a one-dimensional topology.  This is how the
ones from the paper work (they are very simple -- two layers only).
'''
class IndexHierarchy(Hierarchy):
    
    # Create new hierarchy from representation of desired topology.
    def __init__(self, configList):
        pass


'''
Hierarchies in which mid/upper units operate on the entire activation vectors
from the units below them.
'''
class VectorHierarchy(Hierarchy):

    # Create new hierarchy from representation of desired topology.
    #    @param quantize - if True, regularize activation vectors to 0-1;
    #                        else pass activation vectors unmodified
    def __init__(self, configList, quantize=False):
        pass











