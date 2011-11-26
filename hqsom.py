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
    @param inSize - size of input
    @param outSize - size of output
    @param alphaSOM - learning rate for SOM component
    @param alphaRSOM - learning rate for RSOM component
    @parma gamma - time decay rate for RSOM component
    @param initMapUnits - specify if fewer than outSize are desired
    @param grow - grow internal map using Luttrell's method; requires support
                    by underlying SOM/RSOM implementation
    '''
    def __init__(self, inSize, outSize,
                 alphaSOM, alphaRSOM, gamma,
                 initMapUnits=None, grow=False):
        self.som = SOM()
        self.rsom = RSOM()





'''
Configuration object to create hierarchies of base units
'''
class hierarchyConfig(object):
    def __init__(self):
        pass


'''
Assumes the input field is square, that the 

'''
class simple2dHierarchyConfig(object):
    def __init__(self):
        pass





'''
Base class for hierarchies of HQSOM units
'''
class hierarchy(object):
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
class indexHierarchy(hierarchy):
    
    # Create new hierarchy from representation of desired topology.
    def __init__(self, configList):
        pass


'''
Hierarchies in which mid/upper units operate on the entire activation vectors
from the units below them.
'''
class vectorHierarchy(hierarchy):

    # Create new hierarchy from representation of desired topology.
    #    @param quantize - if True, regularize activation vectors to 0-1;
    #                        else pass activation vectors unmodified
    def __init__(self, configList, quantize=False):
        pass











