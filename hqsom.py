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
                 use_pure_implementation=False,  initMapUnits=None, grow=False):
        self.som = SOM(som_input_size, som_map_size, pure=use_pure_implementation)
        self.rsom = RSOM(som_map_size, rsom_map_size, pure=use_pure_implementation)

    '''
    @param unit_input - the test data
    @param gamma_som - the learning rate of the som
    @param gamma_rsom - the learning rate of the rsom
    @param sigma - the spread of the neighborhood function
    @param alpha - the relative time value of data
    '''
    
    def update(self, unit_input, gamma_som=.3, gamma_rsom=.3, sigma_som=.8,
                        sigma_rsom=.8, alpha=.5):
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
Base class from which all hierarchies inherit
'''
class Hierarchy(object):
    pass


'''
Hard-coded layout for the hierarchy shown in figure 3 of the Draper Lab
HQSOM paper.  Bottom layer has 3x3 base units over a 7x7-px (1-px-overlap)
input image, then presents the BMUs of the bottom-layer SOMs in a 1-dimensional
topology to the 2nd-level HQSOM base unit.  Final output is a compressed
invariant representation of the bottom-level input image.
'''
class PaperFig3Hierarchy(Hierarchy):
    
    def __init__(self, som_map_size_bottom, rsom_map_size_bottom,
                       som_map_size_top, rsom_map_size_top,
                       use_pure_implementation = False ):
        self.bottom_hqsom_list = [HQSOM(9, 
                                        som_map_size_bottom,
                                        rsom_map_size_bottom,
                                        use_pure_implementation = use_pure_implementation)
                                  for i in range(9)]
        self.top_hqsom = HQSOM(9, 
                               som_map_size_top, 
                               rsom_map_size_top, 
                               use_pure_implementation = use_pure_implementation)
    
    '''
    Pass in the 7x7-pixel image as a 1-dimensional np.ndarray, enumerated
        through each successive row.
    The other params are just like those for update in the HQSOM base units.
    '''
    def update(self, full_input,
               gamma_som_bottom=.3, gamma_rsom_bottom=.3,
               sigma_som_bottom=.8, sigma_rsom_bottom=.8, alpha_bottom=.5,
               gamma_som_top=.3, gamma_rsom_top=.3,
               sigma_som_top=.8, sigma_rsom_top=.8, alpha_top=.5):
        #print "Training on {}".format(full_input)
        # slice up input image and use it to update the layer-1 HQSOM base units
        bottom_outputs = np.zeros(9)
        shaped = full_input.reshape((7,7))
        for i in range(9):
            row = i % 3
            col = (i-row)/3
            # unit_input is an np.ndarray
            #unit_input = full_input[(2*row+14*col+0):(2*row+14*col+3)]
            ## TODO: these are inefficiently copying the array
            #unit_input = np.append(unit_input, full_input[(2*row+14*col+7):(2*row+14*col+10)])
            #unit_input = np.append(unit_input, full_input[(2*row+14*col+14):(2*row+14*col+17)])
            
            unit_input = shaped[2*col:2*col+3, 2*row:2*row+3].reshape(9,)
            self.bottom_hqsom_list[i].update(unit_input, 
                                             gamma_som_bottom, 
                                             gamma_rsom_bottom,
                                             sigma_som_bottom, 
                                             sigma_rsom_bottom, 
                                             alpha_bottom)
            bottom_outputs[i]= self.bottom_hqsom_list[i].activation_vector(unit_input)
        
        # use outputs from layer-1 HQSOM units to update top-level one
        self.top_hqsom.update(np.array(bottom_outputs),
                                gamma_som_top, gamma_rsom_top,
                                sigma_som_top, sigma_rsom_top, alpha_top)
    
    '''
    this retrieves the output of the top-level hqsom base unit for a given
    7x7 pixel input image.
    
    @param continuous_internal - pass activation vectors internally instead of
                                 just the BMUs (currently unsupported)
    @param continuous_output - retrieve the full activation vector from the
                                 top unit instead of just the BMU
    '''
    def activation_vector(self, full_input, 
                          continuous_internal=False,
                          continuous_output=False):
        
        # slice up input image and get outputs from layer-1 HQSOM base units
        bottom_outputs = np.zeros(9)
        shaped = full_input.reshape((7,7))

        for i in range(9):
            row = i % 3
            col = (i-row)/3
            
            ## unit_input is an np.ndarray
            #unit_input = full_input[(2*row+14*col+0):(2*row+14*col+3)]
            ## TODO: these are inefficiently copying the array
            #unit_input = np.append(unit_input, full_input[(2*row+14*col+7):(2*row+14*col+10)])
            #unit_input = np.append(unit_input, full_input[(2*row+14*col+14):(2*row+14*col+17)])
            
            unit_input = shaped[2*col:2*col+3, 2*row:2*row+3].reshape(9,)

            bottom_outputs[i]= self.bottom_hqsom_list[i].activation_vector(unit_input)
        
        # use outputs from layer-1 HQSOM units to get output from top-level one
        return self.top_hqsom.activation_vector(bottom_outputs, continuous_output)

'''
Configuration object to create hierarchies along one dimension
'''
class HierarchyConfig(object):
    def __init__(self, *args):
        pass


'''
Note - assumes the input field is square
'''
class Simple2dHierarchyConfig(object):
    def __init__(self):
        pass



class NaiveAudioClassifier(Hierarchy):
    
    def __init__(self, som_map_size_bottom, 
                       rsom_map_size_bottom,
                       som_map_size_top, 
                       rsom_map_size_top,
                       use_pure_implementation = False ):
        self.bottom_hqsom = HQSOM(128, 
                                  som_map_size_bottom,
                                  rsom_map_size_bottom,
                                  use_pure_implementation = use_pure_implementation)
        self.top_hqsom = HQSOM(rsom_map_size_bottom, 
                               som_map_size_top,
                               rsom_map_size_top,
                               use_pure_implementation = use_pure_implementation)
                                  
    '''
    Pass in the 7x7-pixel image as a 1-dimensional np.ndarray, enumerated
        through each successive row.
    The other params are just like those for update in the HQSOM base units.
    '''
    def update(self, unit_input,
               gamma_som_bottom=.3, gamma_rsom_bottom=.3,
               sigma_som_bottom=.8, sigma_rsom_bottom=.8, alpha_bottom=.5,
               gamma_som_top=.3, gamma_rsom_top=.3,
               sigma_som_top=.8, sigma_rsom_top=.8, alpha_top=.5):
        self.bottom_hqsom.update(unit_input, 
                                 gamma_som_bottom, 
                                 gamma_rsom_bottom,
                                 sigma_som_bottom, 
                                 sigma_rsom_bottom, 
                                 alpha_bottom)
        bottom_output = self.bottom_hqsom.activation_vector(unit_input,True)
        
        # use outputs from layer-1 HQSOM units to update top-level one
        self.top_hqsom.update(bottom_output,
                              gamma_som_top, 
                              gamma_rsom_top,
                              sigma_som_top, 
                              sigma_rsom_top, 
                              alpha_top)
    
    '''
    this retrieves the output of the top-level hqsom base unit for a given
    7x7 pixel input image.
    
    @param continuous_internal - pass activation vectors internally instead of
                                 just the BMUs (currently unsupported)
    @param continuous_output - retrieve the full activation vector from the
                                 top unit instead of just the BMU
    '''
    def activation_vector(self, unit_input, 
                          continuous_internal=False,
                          continuous_output=False):
        
        # slice up input image and get outputs from layer-1 HQSOM base units
        bottom_output = self.bottom_hqsom.activation_vector(unit_input,True)
        # use outputs from layer-1 HQSOM units to get output from top-level one
        return self.top_hqsom.activation_vector(bottom_output, continuous_output)




'''
PROBABLY DEPRECATED
'''


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











