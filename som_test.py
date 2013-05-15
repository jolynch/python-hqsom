from som import *
from rsom import *
from hqsom import *
from hqsom_audio import *
from preproc.images import *
import preproc.audio as audio
import getopt, sys
import traceback
#import matplotlib.pyplot as plt
import pickle
import genetic_algo


tests = ("som","rsom", "hqsom", "hqsom_noise", "hqsom_noise_multiple", "image_gen", "hqsom_77_network", "hqsom_77", "audio")
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
use_pure = False


input_vectors = np.array([
        [0.1 , 0.1 , 0.1 , 0.1],
        [.01 ,.001 , 0.6 , 0.8 ],
        [0.3 , 0.3 , 0.3 , 0.3],
        [0.0 , 0.8 , 0.0 , 0.0],
        [1.0 , 0.9 , 0.95, 0.82],
        [0.35,0.95 , 0.24, 0.76]])
rate, spread, size, input_size = .4, .2, len(input_vectors), len(input_vectors[0])

def test_som():
    som1 = SOM(input_size, size, pure=use_pure)
    assert som1
    
    #Test that a single vector can be trained on
    print "-- Training on single input --"
    for i in range(10):
        som1.update(input_vectors[0], rate, spread)
    print "Got MSE of {}".format(som1.mse(input_vectors[0]))
    assert som1.mse(input_vectors[0]) < 1e-3
    
    #Test that all vectors can be trained on in a 1:1 network input_size = size
    som1 = SOM(input_size, size, pure=use_pure)
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

    #For paper, disregard
    #data = np.transpose(np.array([
        #[.3 , .7 , .1  , .14 , .01],
        #[.3 , .1 , .01 , .16 , .9],
        #[.3 , .03 , .8 , .7  , .01]]))
    #som1 = SOM(3,5,True)
    #som1.units = data
    #som1.update(np.array((.1,.1,.1)), .2, 1)
    #print som1.units.transpose()
#I'm kind of unsure how to really test this ...
def test_rsom():
    rsom1 = RSOM(input_size, size, pure=use_pure)
    alpha = .3
    #Test a time dependent sequence 
    print "-- Training on alternating values --"
    for i in range(1000):
        rsom1.update(input_vectors[i%2], rate, spread, alpha)
    rsom1.update(input_vectors[2], rate, spread, alpha)
    rsom1.reset()
    assert rsom1.differences[0][0] == 0
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
    hqsom = HQSOM(9,18,3, use_pure_implementation=use_pure)
    def flush(num):
        for l in range(num):
            hqsom.update(test_data[0], g1,g2,s1,s2,a)
    num_cycles, num_repeats = 25, 11
    total_run_count, seq_count = num_cycles*num_repeats*9, 0 
    for j in range(num_cycles):
        for i in range(num_repeats):
            print "update {}/{}".format(seq_count,total_run_count)
            hqsom.reset()
            seq = ()
            if i %2 == 0:
                seq = (1,2,3,1,2,3)
            else:
                seq = (4,5,6,4,5,6)
            for k in seq:
                hqsom.update(test_data[k], g1, g2, s1, s2, a)
            hqsom.reset()
            seq_count += 9

    c = [hqsom.activation_vector(t) for t in test_data]
    print c
    assert c[0] != c[1] and c[1] != c[4]
    assert c[1] == c[2] and c[2] == c[3]
    assert c[4] == c[5] and c[5] == c[6]
    assert c[3] != c[4]



def test_hqsom_noise(noise_std=.1):
    test_data = np.array([
        [0,0,0,0,0,0,0,0,0],
        [1,1,1,0,0,0,0,0,0],
        [0,0,0,1,1,1,0,0,0],
        [0,0,0,0,0,0,1,1,1],
        [1,0,0,1,0,0,1,0,0],
        [0,1,0,0,1,0,0,1,0],
        [0,0,1,0,0,1,0,0,1]])
    #Add in gausian noise
    noise = np.random.normal(0.0,noise_std,test_data.shape)
    test_data = test_data + noise 
    g1,g2,s1,s2,a = .1,.1,16,90,.1
    #Due to the noise we have to add many more map units
    hqsom = HQSOM(9,18,3, use_pure_implementation=use_pure)
    print "bleh"
    def flush(num):
        for l in range(num):
            hqsom.update(test_data[0], g1,g2,s1,s2,a)
    num_cycles, num_repeats = 25, 11
    total_run_count, seq_count = num_cycles*num_repeats*9, 0 
    for j in range(num_cycles):
        for i in range(num_repeats):
            print "update {}/{}".format(seq_count,total_run_count)
            hqsom.reset()
            if i %2 == 0:
                seq = (1,2,3,1,2,3)
            else:
                seq = (4,5,6,4,5,6)
            for k in seq:
                hqsom.update(test_data[k], g1, g2, s1, s2, a)
            hqsom.reset()
            seq_count += 9

    #Re-do the test data to test on different noisy data
    print "genetic_algo.Generating different test data for activating"
    test_data = np.array([
        [0,0,0,0,0,0,0,0,0],
        [1,1,1,0,0,0,0,0,0],
        [0,0,0,1,1,1,0,0,0],
        [0,0,0,0,0,0,1,1,1],
        [1,0,0,1,0,0,1,0,0],
        [0,1,0,0,1,0,0,1,0],
        [0,0,1,0,0,1,0,0,1]])
    #Add in gausian noise
    noise = np.random.normal(0.0,noise_std,test_data.shape)
    test_data = test_data + noise 
    g1,g2,s1,s2,a = .1,.1,16,90,.1
            
    c = [hqsom.activation_vector(t) for t in test_data]
    print c
    assert c[0] != c[1] and c[1] != c[4]
    assert c[1] == c[2] and c[2] == c[3]
    assert c[4] == c[5] and c[5] == c[6]
    assert c[3] != c[4]

def test_hqsom_noise_multiple():
    num_errors, num_tests, noise_std = 0, 100, .2
    for i in range(num_tests):
        try:
            test_hqsom_noise(noise_std)    
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
    hqsom = PaperFig3Hierarchy(65,17,513,output_size, use_pure_implementation=use_pure)
    g1,g2,g3,g4,s1,s2,s3,s4,a1,a2 = .1,.01,.1,.001, 16.0, 100.0, 4.0, 200.0, .1, .01
    data_image = Square_Image(5,(1,1))
    data = data_image.data()
    hqsom.update(data,g1,g2,s1,s2,a1,g3,g4,s3,s4,a2)
    print hqsom.activation_vector(data,False,True)
    assert hqsom.activation_vector(data) != None
    
    
def test_hqsom_77():
    #genetic_algo.Generate the test sequence, note that we must do a spiral exposure to get the
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
                               #use_pure_implementation = use_pure)
    #g1,g2,g3,g4,s1,s2,s3,s4,a1,a2 = .1,.01,.1,.001, 16.0, 100.0, 4.0, 250.0, .1, .01
    #run_name = "PAPER_RUN_GAUSSIAN_"
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
    bottom_som_size, bottom_rsom_size, top_som_size, output_size = 40, 25, 150, 7
    hqsom = PaperFig3Hierarchy(bottom_som_size,
                               bottom_rsom_size,
                               top_som_size,output_size, 
                               use_pure_implementation=use_pure)
    g1,g2,g3,g4,s1,s2,s3,s4,a1,a2 = 0.1,0.01,0.1,0.05,20.0,150.0,15.0,250.0,0.1,0.02    
    run_name = "REFERENCE_19_OUR_SETTINGS_"
    num_cycles, data_sets, num_repeats = 50, [("SQUARE",square_data), ("DIAMOND",diamond_data), ("X",x_data)], 4
    
    seq_num = 0
    
    MAP_Image(hqsom.top_hqsom.rsom.units, "output/{}INITIAL_TOP_RSOM_".format(run_name)).save()
    total_run_count = num_cycles * len(data_sets)*(len(data_sets[0][1])*num_repeats)
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
            #Instead of training on blank data
            print "Resetting SOMS"
            hqsom.reset()
            MAP_Image(hqsom.top_hqsom.rsom.units,"output/{}TOP_RSOM_{}_{}".format(run_name,i,data_type)).save() 
            #for d in blank_data:
                #hqsom.update(d,g1,g2,s1,s2,a1,g3,g4,s3,s4,a2)
                #print "{} update {}/{}".format(data_type, seq_num, total_run_count)
                #print "{} current BMU: {}".format(data_type, hqsom.activation_vector(d))
                #seq_num += 1
    
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
def test_audio(hqsom=None):
    print "Loading songs into memory"
    song_rock = audio.Spectrogram("data/music/Californication.wav")
    song_rock2 = audio.Spectrogram("data/music/ByWay.wav")
    song_techno = audio.Spectrogram("data/music/Everybody.wav")
    song_techno2 = audio.Spectrogram("data/music/DayNNight.wav")
    song_classical = audio.Spectrogram("data/music/Bells.wav")
    song_classical2 = audio.Spectrogram("data/music/Symp9.wav")

    print "Done loading songs into memory"
    songs = [
                ("Techno", song_techno), 
                ("TechnoTEST", song_techno2),
                ("Classical", song_classical),
                ("ClassicalTEST", song_classical2),
                ("Rock", song_rock), 
                ("RockTEST", song_rock2),
            ]
    song_types = [i for (i,j) in songs]
    num_seconds, test_length  = .1, 10
    #Get num_second second slices of each song, looking to a cache first
    try:
        (n,saved_songs,final_data) = pickle.load(open("cache.p", "rb"))
        if not n == num_seconds or not saved_songs == tuple(song_types):
            raise Exception
        print "Found data in cache, skipping generation"
    except:
        print "genetic_algo.Generating ffts"
        raw_data = dict([(i,None) for i in song_types])
        for (song_type, song_file) in songs:
            print "genetic_algo.Generating data on the fly for {} song".format(song_type)
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
    """ 
    plt.matshow(np.transpose(final_data["Rock"]))
    plt.title("Rock")
    plt.matshow(np.transpose(final_data["Techno"]))
    plt.title("Techno")
    plt.matshow(np.transpose(final_data["Classical"]))
    plt.title("Classical")
    plt.matshow(np.transpose(final_data["ClassicalTEST"]))
    plt.title("Classical_TEST_DATA")
    plt.matshow(np.transpose(final_data["TechnoTEST"]))
    plt.title("Techno_TEST_DATA")
    plt.matshow(np.transpose(final_data["RockTEST"]))
    plt.title("Rock_TEST_DATA")
    """  
    output_size = 5
    if hqsom is None:
        hqsom = Hierarchy1D(
            LayerConf1D(2, 64, 128, 0,
                        50, 0.2, 200,
                        40, .7, 0.15, 100, use_pure),
            LayerConf1D(2, 1, 2, 0,
                        50, 0.2, 200,
                        20, .7, 0.15, 100, use_pure),
            LayerConf1D(1, 2, 2, 0,
                        32, 0.2, 200,
                        output_size, .05, 0.2, 100, use_pure),
        )
    #hqsom = NaiveAudioClassifier(bottom_som_size,
                               #bottom_rsom_size,
                               #top_som_size,output_size, 
                               #use_pure_implementation = True)
    #hqsom = genetic_algo.Genome(128, output_size).to_hierarchy()
    #genome = genetic_algo.Genome(128, 5, [genetic_algo.Gene(128, 1, [128, 1, 0.5349470927446156, 58, 0.16262059789324113, 93, 69, 0.38946495945845583, 0.18591242958088183, 449]),
    #                         genetic_algo.Gene(1, 1, [1, 1, 0.9697823529658623, 67, 0.06338912516811035, 484, 5, 0.07069243885373111, 0.30821633466399, 312])])

    #genome = genetic_algo.Genome(128, 5, [
    #    genetic_algo.Gene(128, 1, [128, 1, 0.8191182230079156, 86, 0.13323972043189236, 175, 31, 0.3806979377580392, 0.8121811036319838, 98]),
    #    genetic_algo.Gene(1, 1, [1, 1, 0.8727135450401478, 62, 0.3453597203536144, 121, 50, 0.755878448191539, 0.6818380459687157, 325]),
    #    genetic_algo.Gene(1, 1, [1, 1, 0.4174074007331876, 89, 0.7549203282530946, 50, 5, 0.7849685525193116, 0.5789786448249847, 263])
    #    ])
    #hqsom = genome.to_hierarchy()
    print hqsom.layer_configs
    run_name = "AUDIO_TEST"
           
    #Testing schema:
    # 1) Expose to entirety of three songs
    # 2) Pick 3 random sequences of test_length in size from each song, run through
    # 3) Clear at each in between
    seq_num = 0
    num_cycles, num_repeats = 1, 1
    total_run_count = num_cycles*sum([(len(final_data[x])) for x in song_types])

    for i in range(num_cycles):
        for song_type in song_types:
            if song_type == "ClassicalTEST" or song_type == "TechnoTEST" or song_type == "RockTEST":
                print "Skipping test data: {}".format(song_type)
                continue
            for spectrum in final_data[song_type]:
                hqsom.update(spectrum)
                #print hqsom.activation_vector(spectrum, True, True)
                print "{} update {}/{}".format(song_type, seq_num, total_run_count)
                seq_num += 1
            print "Resetting RSOMs"
            hqsom.reset()
    
    total_run_count = num_cycles*2*len(song_types)*test_length
    seq_num = 0
    for i in range(num_cycles*2):
        for song_type in song_types:
            if song_type == "ClassicalTEST" or song_type == "TechnoTEST" or song_type == "RockTEST":
                print "Skipping test data: {}".format(song_type)
                continue
            num_spectrograms = len(final_data[song_type])
            r_index = np.random.randint(0,num_spectrograms-test_length)
            for index in range(r_index, r_index+test_length):
                hqsom.update(final_data[song_type][index])
                #print hqsom.activation_vector(spectrum, False, True)
                print "{} update {}/{}".format(song_type, seq_num, total_run_count)
                seq_num += 1
            print "Resetting RSOMs"
            hqsom.reset()
                
    
             
    print "Run: {}".format(run_name)
    print "Using Network:"
    print hqsom.layer_configs

    print "num_cycles, num_repeats, num_seconds, test_length = {}, {}, {}, {}".format(num_cycles, num_repeats, num_seconds, test_length)
    
    for data_name in song_types:
        print "#"*80
        print "Results for {}".format(data_name)
        data_collection = final_data[data_name]
        results =[0]*output_size
        for spect in data_collection:
            results[hqsom.activation_vector(spect)] += 1
        t = sum(results)
        results = [float(i)/t for i in results]
        results = np.array(results)
        print "Final Distribution Over Map Space"
        print results
        print "MODE: {}".format(np.argmax(results))
    #plt.show()

    
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


