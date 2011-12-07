'''
Networks for audio classification

As a goal, we want to train a network to classify sections from
particular songs.

Procedure:

1. Grab three songs; isolate 30 seconds; convert to WAV.

2. Arbitrarily select 5 seconds from each song to be out-of-sample data.

3. Select a network from one of the below designs.

4. Train the network on the in-sample data using the audio train/test framework.
    i. choose a song
    ii. choose a spectrogram FFT window size, ideally pretty long
    iii. generate spectrograms for each of the in-sample sections
    iv. for each song, feed the entire spectrogram to the network, using
            high gamma and sigma
        a. reset the network at all levels when finished with a given song.
                i.e. reset the rsom EMA to zero.
        b. cycle through all songs a few times
    v. if your top-level node is not constant, you're doing it wrong
    vi. make sigma smaller, but leave gamma fairly high
    vii. rotate through songs 5 seconds or something at a time instead of entire
            song.  reset network between songs.
    viii. make sigma smaller and gamma smaller
    ix. rotate through very short sections of songs, and cycle between them
            A LARGE NUMBER OF TIMES.
    x. if your top-level node does not change to a different constant shortly
            after switching to a new song, you're likewise doing it wrong

5. Test classification performance
    i. choose a song
    ii. generate spectrograms of the out-of-sample snippets using the
                same window size as in training above
    iii. feed in each spectrogram and see if the top-level node produces the
                same output as it did during training for this song

'''


'''
Network characteristics:

We want a multi-layered network, because we are targeting highly invariant
pattern recognition.

We want a slow (small) alpha given how long these sequences are and how much
invariance we want at the temporal level.

We want a slow gamma for similar reasons -- don't want to thrash nodes around
in map space too much.  This is more relevant at the higher levels of the
network.

Lower levels should have higher gammas and huge SOM map spaces m_s to learn
particular chords that tend to show up a lot in the spectrogram.  Lower-level
RSOM map spaces should also be decently large, but nowhere near m_s^2.  These
learn short chord progressions.

Mid-levels should be medium for these things.  They learn a smaller number of
longer and more general chord progressions.

Top levels should be very highly restricted, especially the RSOM map size, so
that only the most general features of a song are allowed representation.

Specific tweaking will be needed, of course.  These are just general thoughts.


Also, spatial aspects are likely to be more important that spatiotemporal ones.

I.e. we care more about specific chords than progressions etc.

Frankly, we could do a decent job of prediction from just a decision tree on
the chords....  And we should write this in the paper.  The point of the HQSOM
architecture is to build a very very general unsupervised learner, and we see
the costs of that generality reflected as complexity and inefficiency.
'''


from hqsom import *





# This is just a configuration object for layers in a 1-dimensional hierarchy
class LayerConf1D(object):
    '''
    @param numHQSOMs - how many nodes in this layer
    @param node_inputs - how many inputs per node
    @param total_inputs - how many total inputs to the layer
    @param overlap - how much overlap you want between inputs for nodes in this
                     layer, as a number of inputs to overlap between any two
                     adjacent nodes.  Yes, this is overdetermined.  It's for
                     the sake of clarity when using it.
    
    Example: Suppose I want 40 total inputs, 3 nodes taking 20 inputs each,
                overlap of 10 between adjacent pairs.  Node 1 gets inputs 1-20;
                node 2 gets inputs 11-30; node 3 gets inputs 21-40.
    
    @param m_s - SOM map size for each node in layer
    @param gamma_s - SOM gamma - learning rate
    @param sigma_s - SOM sigma - learning region size scale factor
    
    @param m_r - RSOM map size for each node
    @param alpha_r - RSOM alpha - temporal leaky integrator leak rate
    @param gamma_r - RSOM gamma - see above
    @param sigma_r - RSOM sigma - see above
    '''
    def __init__(self, numHQSOMs, node_inputs, total_inputs, overlap,
                 m_s, gamma_s, sigma_s,
                 m_r, alpha_r, gamma_r, sigma_r,
                use_pure_implementation = False):
        self.numHQSOMs = numHQSOMs
        self.node_inputs = node_inputs
        self.total_inputs = total_inputs
        self.overlap = overlap
        self.m_s = m_s
        self.gamma_s = gamma_s
        self.sigma_s = sigma_s
        self.m_r = m_r
        self.alpha_r = alpha_r
        self.gamma_r = gamma_r
        self.sigma_r = sigma_r
        self.use_pure_implementation = use_pure_implementation
    
    def __repr__(self):
        s = ""
        for attr, value in self.__dict__.iteritems():
            s += "{} => {} \n".format(attr, value)
        return s





# This is an actual hierarchy, configured by LayerConfig1D instances
# Intended for use with audio.
class Hierarchy1D(Hierarchy):
    
    # Pass in the Layer1D objects representing the layers you want, in order
    # from bottom layer to top layer.
    def __init__(self, *args):
        # make layers
        self.layer_configs = args
        self.layers = []
        assert args[-1].numHQSOMs == 1, "top layer must contain only one unit"
        # for verifying that each layer has a number of inputs equal to the
        #        number of nodes in the layer underneath
        last_num_nodes = args[0].total_inputs
        for layer_cf in args:
            assert last_num_nodes == layer_cf.total_inputs, "input size of a "+\
                "given layer must match number of nodes in the layer below it"
            self.layers.append([])
            # make each node within a given layer
            for i in range(layer_cf.numHQSOMs):
                node = HQSOM(layer_cf.node_inputs, layer_cf.m_s, layer_cf.m_r,
                            use_pure_implementation =
                                        layer_cf.use_pure_implementation)
                self.layers[-1].append(node)
            last_num_nodes = layer_cf.numHQSOMs
#        print self.layer_configs
#        print self.layers
    
    
    # Use this when you want to start using new settings.  This allows you to
    # avoid having to pass in every param for every layer during every update.
    #         
    # @param layer - these are 0-based, indexed from bottom to top
    def specify_new_layer_params(self, layer, gamma_s, sigma_s,
                                 alpha_r, gamma_r, sigma_r):
        cf = self.layer_configs[layer]
        cf.gamma_s = gamma_s
        cf.sigma_s = sigma_s
        cf.alpha_r = alpha_r
        cf.gamma_r = gamma_r
        cf.sigma_r = sigma_r
        
        
    # This is much simpler than the figure 3 hierarchy's update method signature
    def update(self, full_input, verbose=False):
        #print "Training on {}".format(full_input)
        if verbose:
            print "Following is full input"
            print full_input
        prev_layer_output = full_input
        next_layer_input = []
        for i in range(len(self.layers)):
            if verbose:
                print "#"*80
                print "Entering layer {}".format(i)
            layer = self.layers[i]
            layer_cf = self.layer_configs[i]
            for j in range(len(layer)):
                node = layer[j]
                start_idx = j*(layer_cf.node_inputs-layer_cf.overlap)
                node.update(prev_layer_output[start_idx : start_idx+layer_cf.node_inputs],
                            layer_cf.gamma_s, layer_cf.gamma_r,
                            layer_cf.sigma_s, layer_cf.sigma_r,
                            layer_cf.alpha_r)
                next_layer_input.append(
                            node.activation_vector(
                                prev_layer_output[
                                  start_idx : start_idx+layer_cf.node_inputs ]))
            prev_layer_output = next_layer_input
            if verbose:
                print "Passing {} to next layer".format(prev_layer_output)
            next_layer_input = []

    
    '''
    this retrieves the output of the top-level hqsom base unit
    
    @param continuous_internal - pass activation vectors internally instead of
                                 just the BMUs (currently unsupported)
    @param continuous_output - retrieve the full activation vector from the
                                 top unit instead of just the BMU
    '''
    def activation_vector(self, full_input, 
                          continuous_internal=False,
                          continuous_output=False,
                          verbose = False):
        # keep track of input to a given layer and output to be fed to the
        #         layer above it
        prev_layer_output = full_input
        next_layer_input = []
        # iterate through layers, propagating information upward
        for i in range(len(self.layers)):
            layer = self.layers[i]
            layer_cf = self.layer_configs[i]
            # if we're not at the top layer yet
            if i < len(self.layers)-1:
                # iterate through nodes, feeding a chunk of the layer input to each
                for j in range(len(layer)):
                    node = layer[j]
                    start_idx = j*(layer_cf.node_inputs-layer_cf.overlap)
                    next_layer_input.append(
                                node.activation_vector(
                                    prev_layer_output[
                                      start_idx : start_idx+layer_cf.node_inputs ]))
            else: # we're at the top layer, a single node
                return layer[0].activation_vector(prev_layer_output, continuous_output)
            prev_layer_output = next_layer_input
            if verbose:
                print "Prev layer input for layer: {}".format(i)
                print prev_layer_output
            next_layer_input = []

    # reset the RSOM EMAs of all nodes.  this allows us to prevent spurious
    # associations between songs when switching to a new song.
    def reset(self):
        for layer in self.layers:
            for node in layer:
                node.reset()




if __name__ == "__main__":
    
    
    '''
    basic testing
    '''
    
    # make a hierarchy with 128 inputs:
    test_hierarchy_1 = Hierarchy1D(
           # layer 1: 16 nodes over 8 inputs each
           LayerConf1D(16, 8, 128, 0,
                 64, 0.5, 100.0,
                 32, 0.2, 0.1, 200.0),
           # layer 2: 4 nodes over 4 inputs each
           LayerConf1D(4, 4, 16, 0,
                 256, 0.3, 100.0,
                 256, 0.03, 0.05, 200.0),
           # layer 3: 1 node over the 4 in layer 2
           LayerConf1D(1, 4, 4, 0,
                 64, 0.1, 100.0,
                 32, 0.01, 0.02, 200.0))
#    def __init__(self, numHQSOMs, node_inputs, total_inputs, overlap,
#             m_s, gamma_s, sigma_s,
#             m_r, alpha_r, gamma_r, sigma_r,
#             use_pure_implementation = False):
    
    bogus_data = range(128)
    
    
    for i in range(50):
        print "UPDATING: round " + str(i)
        test_hierarchy_1.update(bogus_data)
        print "\n\n\n"
    
    print test_hierarchy_1.activation_vector(bogus_data, False, True)
    
    
    
    
    
    
    
















































