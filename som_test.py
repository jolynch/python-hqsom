from som import *
from hqsom import *
from hqsom_audio import *
from preproc.images import *
import preproc.audio as audio
import getopt, sys
import traceback
import matplotlib.pyplot as plt
import pickle


tests = ("som","rsom", "hqsom", "hqsom_noise", "hqsom_noise_multiple", "image_gen", "hqsom_77_network", "hqsom_77", "audio")
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


input_vectors = np.array([
        [0.1 , 0.1 , 0.1 , 0.1],
        [.01 ,.001 , 0.6 , 0.8 ],
        [0.3 , 0.3 , 0.3 , 0.3],
        [0.0 , 0.8 , 0.0 , 0.0],
        [1.0 , 0.9 , 0.95, 0.82],
        [0.35,0.95 , 0.24, 0.76]])
rate, spread, size, input_size = .4, .2, len(input_vectors), len(input_vectors[0])

def test_som():
    som1 = SOM(input_size, size)
    assert som1
    
    #Test that a single vector can be trained on
    print "-- Training on single input --"
    for i in range(10):
        som1.update(input_vectors[0], rate, spread)
    print "Got MSE of {}".format(som1.mse(input_vectors[0]))
    assert som1.mse(input_vectors[0]) < 1e-3
    
    #Test that all vectors can be trained on in a 1:1 network input_size = size
    som1 = SOM(input_size, size)
    print "-- Training on all inputs --"
    for i in range(1000):
        som1.update(input_vectors[i%len(input_vectors)], rate, spread)
   
    total_mse = 0
    for inp in input_vectors:
        total_mse += som1.mse(inp)
        print "Got MSE of {}".format(som1.mse(inp))
        assert som1.mse(inp) < .3
    assert total_mse < .05 * len(input_vectors)
    #Checking Activation vectors
    activated = set()
    for inp in input_vectors:
        activated.add(som1.bmu(inp))
        print "Applying signal: {}".format(inp)
        print "Activating {}".format(som1.units[som1.bmu(inp)])
    err = abs(len(activated) - len(input_vectors))
    print activated
    print "All activated units: {}".format(activated)
    print "Error: {} vs max {}".format(err, .5*len(input_vectors))
    assert err <= .5*len(input_vectors)

#I'm kind of unsure how to really test this ...
def test_rsom():
    rsom1 = RSOM(input_size, size)
    alpha = .3
    #Test a time dependent sequence 
    print "-- Training on alternating values --"
    for i in range(1000):
        rsom1.update(input_vectors[i%2], rate, spread, alpha)
    rsom1.update(input_vectors[2], rate, spread, alpha)
    for i in range(3):
        print "Got MSE of {}".format(rsom1.mse(input_vectors[i]))
        print "Activation vector: {}".format(rsom1.activation_vector(input_vectors[i], True))
        assert rsom1.mse(input_vectors[i%2]) < .3 

def test_hqsom():
    test_data = np.array([
        [0,0,0,0,0,0,0,0,0],
        [1,1,1,0,0,0,0,0,0],
        [0,0,0,1,1,1,0,0,0],
        [0,0,0,0,0,0,1,1,1],
        [1,0,0,1,0,0,1,0,0],
        [0,1,0,0,1,0,0,1,0],
        [0,0,1,0,0,1,0,0,1]])
    
    g1,g2,s1,s2,a = .1,.1,16,90,.1
    hqsom = HQSOM(9,18,3, use_pure_implementation=True)
    def flush(num):
        for l in range(num):
            hqsom.update(test_data[0], g1,g2,s1,s2,a)
    num_cycles, num_repeats = 25, 11
    total_run_count, seq_count = num_cycles*num_repeats*9, 0 
    for j in range(num_cycles):
        for i in range(num_repeats):
            print "update {}/{}".format(seq_count,total_run_count)
            flush(3)
            seq = ()
            if i %2 == 0:
                seq = (1,2,3)
            else:
                seq = (4,5,6)
            for k in seq:
                hqsom.update(test_data[k], g1, g2, s1, s2, a)
            flush(3)
            seq_count += 9

    c = [hqsom.activation_vector(t) for t in test_data]
    print c
    assert c[0] != c[1] and c[1] != c[4]
    assert c[1] == c[2] and c[2] == c[3]
    assert c[4] == c[5] and c[5] == c[6]
    assert c[3] != c[4]



def test_hqsom_noise():
    test_data = np.array([
        [0,0,0,0,0,0,0,0,0],
        [1,1,1,0,0,0,0,0,0],
        [0,0,0,1,1,1,0,0,0],
        [0,0,0,0,0,0,1,1,1],
        [1,0,0,1,0,0,1,0,0],
        [0,1,0,0,1,0,0,1,0],
        [0,0,1,0,0,1,0,0,1]])
    #Add in gausian noise
    noise = np.random.normal(0.0,.05,test_data.shape)
    test_data = test_data + noise 
    g1,g2,s1,s2,a = .1,.1,16,90,.1
    #Due to the noise we have to add many more map units
    hqsom = HQSOM(9,45,3, use_pure_implementation=True)
    def flush(num):
        for l in range(num):
            hqsom.update(test_data[0], g1,g2,s1,s2,a)
    num_cycles, num_repeats = 45, 11
    total_run_count, seq_count = num_cycles*num_repeats*9, 0 
    for j in range(num_cycles):
        for i in range(num_repeats):
            print "update {}/{}".format(seq_count,total_run_count)
            flush(3)
            seq = ()
            if i %2 == 0:
                seq = (1,2,3)
            else:
                seq = (4,5,6)
            for k in seq:
                hqsom.update(test_data[k], g1, g2, s1, s2, a)
            flush(3)
            seq_count += 9

    c = [hqsom.activation_vector(t) for t in test_data]
    print c
    assert c[0] != c[1] and c[1] != c[4]
    assert c[1] == c[2] and c[2] == c[3]
    assert c[4] == c[5] and c[5] == c[6]
    assert c[3] != c[4]

def test_hqsom_noise_multiple():
    num_errors, num_tests = 0, 5
    np.random.seed()
    for i in range(num_tests):
        try:
            test_hqsom_noise()    
        except:
            num_errors += 1
    print "Passed {} out of {}".format(num_tests-num_errors, num_tests)
    assert num_errors < .25 * num_tests

def enumerate_spiral(l):
    coords, coord, original_l = [], [0,0], l
    while l > 0:
        #Go down
        for i in range(l):
            if not tuple(coord) in coords:
                coords.append(tuple(coord))
            coord[1]+=1
            #print "going down from {} to {}".format(coords[-1], coord)
        if l < original_l:
            l -= 1
        #Go right
        for i in range(l):
            if not tuple(coord) in coords:
                coords.append(tuple(coord))
            coord[0]+=1
            #print "going right from {} to {}".format(coords[-1], coord)
        #Go up
        for i in range(l):
            if not tuple(coord) in coords:
                coords.append(tuple(coord))
            coord[1]-=1
            #print "going up from {} to {}".format(coords[-1], coord)
        l -= 1
        #Go left
        for i in range(l):
            if not tuple(coord) in coords:
                coords.append(tuple(coord))
            coord[0]-=1
            #print "going left from {} to {}".format(coords[-1], coord)
    coords.append(coord)
    return coords

def test_hqsom_77_network():
    output_size =17
    hqsom = PaperFig3Hierarchy(65,17,513,output_size, use_pure_implementation = True)
    g1,g2,g3,g4,s1,s2,s3,s4,a1,a2 = .1,.01,.1,.001, 16.0, 100.0, 4.0, 200.0, .1, .01
    data_image = Square_Image(5,(1,1))
    data = data_image.data()
    hqsom.update(data,g1,g2,s1,s2,a1,g3,g4,s3,s4,a2)
    print hqsom.activation_vector(data,False,True)
    assert hqsom.activation_vector(data) != None
    
    
def test_hqsom_77():
    #Generate the test sequence, note that we must do a spiral exposure to get the
    #correct temporal-spatial representations in the SOMS
    #7x7 only has one possible test (cycled twice of course)
    coord_test = {"large":[(7,0,0),(7,0,0)]}
    #5x5 has 9 possible positions (cycled twice of course)
    coord_test["medium"] = [(5,i,j) for (i,j) in enumerate_spiral(2)]
    coord_test["medium"] = coord_test["medium"][::-1] + coord_test["medium"] 
    #3x3 has 25 possible positions (cycled twice of course)
    coord_test["small"] = [(3,i,j) for (i,j) in enumerate_spiral(4)]
    coord_test["small"] = coord_test["small"][::-1] + coord_test["small"] 
    #######The available data sets
    square_data, diamond_data, x_data = [], [], []
    #First we spiral out, then back in for each data set
    for data_type,data_class,data_container in [("square", Square_Image, square_data),
                                                ("diamond", Diamond_Image, diamond_data),
                                                ("x", X_Image, x_data)]:
        for data_set in ("large","medium", "small"):
            for (w,x,y) in coord_test[data_set]:
                if data_type == "diamond":
                    w,x,y = w/2, x+w/2, y+w/2
                image_data = data_class(w,(x,y))
                data_container.append(image_data.data())
                image_data.save("data/{}_#{}#_".format(data_type, str(len(data_container)).zfill(2)))
    
    blank_data = [Data_Image().data() for i in range(20)]
    #print len(square_data)
    #print len(diamond_data)
    #print len(x_data)

    #Paper settings
    #Make sure we don't use any of our "improvements"
    #bottom_som_size, bottom_rsom_size, top_som_size, output_size = 65,17,513,17
    #hqsom = PaperFig3Hierarchy(bottom_som_size,
                               #bottom_rsom_size,
                               #top_som_size,output_size, 
                               #use_pure_implementation = True)
    #g1,g2,g3,g4,s1,s2,s3,s4,a1,a2 = .1,.01,.1,.001, 16.0, 100.0, 4.0, 250.0, .1, .01
    #run_name = "PAPER_FEWER_BLANK_"
    #num_cycles, data_sets, num_repeats = 150, [("SQUARE",square_data), ("DIAMOND",diamond_data), ("X",x_data)], 5

    
    #Really good TWO classifier:
    #bottom_som_size, bottom_rsom_size, top_som_size, output_size = 10,80,10,5
    #hqsom = PaperFig3Hierarchy(bottom_som_size,
                               #bottom_rsom_size,
                               #top_som_size,output_size, 
                               #use_pure_implementation = True)
    #g1,g2,g3,g4,s1,s2,s3,s4,a1,a2 = .2,.4,.1,.5, 10.0, 80.0, 14.0, 100.0, .8, .01
    #run_name = "TWO_CLASS_"
    #num_cycles, data_sets, num_repeats = 1, [("SQUARE",square_data), ("DIAMOND",diamond_data), ("X",x_data)], 1

    #Our settings
    bottom_som_size, bottom_rsom_size, top_som_size, output_size = 200, 150, 200, 50
    hqsom = PaperFig3Hierarchy(bottom_som_size,
                               bottom_rsom_size,
                               top_som_size,output_size, 
                               use_pure_implementation = True)
    g1,g2,g3,g4,s1,s2,s3,s4,a1,a2 = 0.4,0.02,0.35,0.05,40.0,150.0,40.0,250.0,0.15,0.1
    run_name = "15_OUR_SETTINGS_"
    num_cycles, data_sets, num_repeats = 75, [("SQUARE",square_data), ("DIAMOND",diamond_data), ("X",x_data)], 5
    
    seq_num = 0
    
    MAP_Image(hqsom.top_hqsom.rsom.units, "output/{}INITIAL_TOP_RSOM_".format(run_name)).save()
    total_run_count = num_cycles * len(data_sets)*(len(data_sets[0][1])*num_repeats+len(blank_data))
    for i in range(num_cycles):
        for data_type, data_set in data_sets:
            for j in range(num_repeats):
                MAP_Image(hqsom.top_hqsom.rsom.units,"output/{}TOP_RSOM_{}_{}_{}".format(run_name,i,data_type,j)).save() 
                for d in data_set:
                    hqsom.update(d,g1,g2,s1,s2,a1,g3,g4,s3,s4,a2)
                    print "{} update {}/{}".format(data_type, seq_num, total_run_count)
                    print "{} current BMU: {}".format(data_type, hqsom.activation_vector(d))
                    seq_num += 1
            data_type = "BLANK"
            MAP_Image(hqsom.top_hqsom.rsom.units,"output/{}TOP_RSOM_{}_{}".format(run_name,i,data_type)).save() 
            for d in blank_data:
                hqsom.update(d,g1,g2,s1,s2,a1,g3,g4,s3,s4,a2)
                print "{} update {}/{}".format(data_type, seq_num, total_run_count)
                print "{} current BMU: {}".format(data_type, hqsom.activation_vector(d))
                seq_num += 1
    
    print "Collecting Classification Data, please wait this can take time"
    data_sets = [("BLANK", blank_data)]+data_sets
    output_hash = {"BLANK":[0]*output_size,"SQUARE":[0]*output_size,"DIAMOND":[0]*output_size,"X":[0]*output_size}
    for data_name, data_collection in data_sets:
        for i in data_collection:
            result = hqsom.activation_vector(i)
            output_hash[data_name][result] += 1
    print "Run: {}".format(run_name)
    print "Using the parameters g1,g2,g3,g4,s1,s2,s3,s4,a1,a2 = {},{},{},{},{},{},{},{},{},{}".format(g1,g2,g3,g4,s1,s2,s3,s4,a1,a2)
    print "Using {} cycles of each data set repeated {} times".format(num_cycles, num_repeats)
    print "BSOM, BRSOM, TSOM, TRSOM sizes: {}, {}, {}, {}".format(bottom_som_size, bottom_rsom_size, top_som_size, output_size)
    for data_name, data_collection in data_sets:
        mode = np.argmax(output_hash[data_name])
        num_items = float(len(data_collection))
        print "#"*80
        print "Data Set: {}".format(data_name)
        print "Most Frequently Classified As (MODE): {}".format(mode)
        results = np.array(output_hash[data_name])
        print "Full Distribution over Final RSOM Map Space:"
        print results / num_items
    MAP_Image(hqsom.bottom_hqsom_list[5].rsom.units,"output/{}FINAL_MIDDLE_RSOM".format(run_name)).save() 

#WE ONLY SUPPORT wave files of the <b>same bitrate</b>
def test_audio():
    print "Loading songs into memory"
    song_rock = audio.Spectrogram("data/music/Californication.wav")
    song_techno = audio.Spectrogram("data/music/Everybody.wav")
    song_classical = audio.Spectrogram("data/music/Brahms_Double_Concerto_in_A_minor_smaller.wav")
    print "Done loading songs into memory"
    songs = [
                ("Rock", song_rock), 
                ("Techno", song_techno), 
                ("Classical", song_classical)
            ]
    song_types = [i for (i,j) in songs]
    num_seconds, test_length  = .1, 4
    #Get num_second second slices of each song, looking to a cache first
    try:
        (n,saved_songs,final_data) = pickle.load(open("cache.p", "rb"))
        if not n == num_seconds or not saved_songs == tuple(song_types):
            raise Exception
        print "Found data in cache, skipping generation"
    except:
        print "Generating ffts"
        raw_data = dict([(i,None) for i in songs])
        for (song_type, song_file) in songs:
            print "Generating data on the fly for {} song".format(song_type)
            fft_length = song_file.sample_rate * num_seconds
            #To get a power of 2
            fft_length = int(2**np.ceil(np.log(fft_length)/np.log(2)));
            print "Using fft_length of {}".format(fft_length)
            raw_data[song_type] = song_file.get_spectrogram(fft_length)
        
        print "Reshaping ffts into length 128 inputs"
        final_data = {}
        for song_type in song_types:
            data = raw_data[song_type]
            new_data = np.zeros((data.shape[0], 128))
            bucket_sum, spect = 0, None
            for spect_index in range(len(data)):
                print "{} of {} Spectrograms processed".format(spect_index, len(data))
                spect = data[spect_index]
                window_size = len(spect) / 128
                bucket_sum = 0
                for i in range(128):
                    #bucket_sum = np.mean(spect[i*window_size:i*window_size+window_size])
                    new_data[spect_index][i] = spect[i*window_size]
                #new_data[spect_index] = new_data[spect_index] - min(new_data[spect_index])
                #new_data[spect_index] = new_data[spect_index] / np.linalg.norm(new_data[spect_index])
                
            final_data[song_type] = new_data 
        pickle.dump((num_seconds, tuple(song_types), final_data), open("cache.p","wb"))
    #plt.matshow(np.transpose(final_data["Rock"]))
    #plt.matshow(np.transpose(final_data["Techno"]))
    #plt.matshow(np.transpose(final_data["Classical"]))
    #plt.show()
    print "DONE generating test data"
    print "Generating training sequences"
    training_seq = dict([(i,[]) for i in song_types])
    for song_type in song_types:
        num_samples = len(final_data[song_type])
        index = np.random.randint(10)
        while index+test_length < num_samples:
            training_seq[song_type].extend(range(index, index+test_length))
            index = index + 2*test_length
    #for key in training_seq:
        #training_seq[key] = training_seq[key]+training_seq[key][::-1]
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
    LayerConf1D (numHQSOMs, node_inputs, total_inputs, overlap,
                 m_s, gamma_s, sigma_s,
                 m_r, alpha_r, gamma_r, sigma_r,
                use_pure_implementation = False):
    '''
    output_size = 32
    hqsom = Hierarchy1D(
        ## layer 1: 8 nodes over 16 inputs each
        #LayerConf1D(8, 16,   128, 0,
                    #40, 0.1, 1.0,
                    #20, 0.4, 0.1, 50.0,True),
        ###layer 2: 1 node over the 8 in layer 2
        #LayerConf1D(1,  8,   8, 0,
                    #40, 0.1, 1.0,
                    #output_size, 0.1, 0.08, 2.0,True))
         
        #Too slow, possibly better
        # layer 1: 16 nodes over 8 inputs each
        # layer 1: 16 nodes over 8 inputs each
        LayerConf1D(16, 8, 128, 0,
                    128, 0.3, 2.0,
                    64, 0.1, 0.1, 1.0, True),
        # layer 2: 4 nodes over 4 inputs each
        LayerConf1D(4, 4, 16, 0,
                    32, 0.15, 1.0,
                    16, 0.03, 0.05, 2.0, True),
        # layer 3: 1 node over the 4 in layer 2
        LayerConf1D(1, 4, 4, 0,
                    32, 0.1, 2.0,
                    output_size, 0.001, 0.02, 4.0, True))
    #hqsom = NaiveAudioClassifier(bottom_som_size,
                               #bottom_rsom_size,
                               #top_som_size,output_size, 
                               #use_pure_implementation = True)
    num_cycles, num_repeats = 4, 1
    run_name = "AUDIO_TEST"
    seq_num = 0
           
    blank = np.zeros(128)
    print song_types
    total_run_count = num_cycles * sum([(len(training_seq[x]))*num_repeats+10 for x in song_types])
    for i in range(num_cycles):
        for data_type in song_types:
            for j in range(num_repeats):
                for spectrum_index in training_seq[data_type]:
                    #print "updating with:"
                    #print final_data[data_type][spectrum_index]
                    hqsom.update(final_data[data_type][spectrum_index])
                    print hqsom.activation_vector(final_data[data_type][spectrum_index], False, True)
                    print "{} update {}/{}".format(data_type, seq_num, total_run_count)
                    seq_num += 1
            data_type = "BLANK"
            for i in range(10):
                hqsom.update(blank)
                print hqsom.activation_vector(blank, False, True)
                print "{} update {}/{}".format(data_type, seq_num, total_run_count)
                    
                seq_num += 1
            
    print "Run: {}".format(run_name)
    print "Using {} cycles of each data set repeated {} times".format(num_cycles, num_repeats)
    song_types = [i for i in song_types] + ["BLANK"]
    final_data["BLANK"] = blank.reshape(1,128)
    for data_name in song_types:
        print "#"*80
        print "Results for {}".format(data_name)
        data_collection = final_data[data_name]
        results =[0]*output_size
        for spect in data_collection:
            results[hqsom.activation_vector(spect)] += 1
            #print hqsom.activation_vector(spect, False, True)
        print "Got: {}".format(results)
        #mode = np.argmax(output_hash[data_name])
        #num_items = float(len(data_collection))
        #print "#"*80
        #print "Data Set: {}".format(data_name)
        #print "Most Frequently Classified As (MODE): {}".format(mode)
        #results = np.array(output_hash[data_name])
        #print "Full Distribution over Final RSOM Map Space:"
        #print results / num_items   

    
if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "t:l", ["list","test="])
    except getopt.GetoptError, err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"

    #So that we get reproduceable results
    np.random.seed(15717)
    for o,a in opts:
        if o in ("-t", "--test"):
            print "Running {} test:".format(a)
            try:
                eval("test_"+a)()
            except Exception as e:
                print e
                traceback.print_exc(file=sys.stdout)
                print "!!! ERROR !!!"
            else:
                print "SUCCESS"
        elif o in ("-l", "--list"):
            print "List of tests: {}".format(tests)
        # print help information and exit:
    if len(opts) == 0:
        print "Running all Tests"
        for test in tests:
            print "#"*80
            print "Running test on: {}".format(test)
            print "-"*80
            try:
                eval("test_"+test)()
            except Exception as e :
                print e
                traceback.print_exc(file=sys.stdout)
                print "!!! ERROR !!!"
            else:
                print "SUCCESS"
            print "#"*80


