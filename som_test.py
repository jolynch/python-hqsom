from som import *
from hqsom import *
from preproc.images import *
import getopt, sys
import traceback

tests = ("som","rsom", "hqsom", "hqsom_noise", "hqsom_noise_multiple", "image_gen", "hqsom_77")

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
    hqsom = HQSOM(9,65,3, use_pure_implementation=True)
    def flush(num):
        for l in range(num):
            hqsom.update(test_data[0], g1,g2,s1,s2,a)

    for j in range(40):
        for i in range(11):
            flush(3)
            seq = ()
            if i %2 == 0:
                seq = (1,2,3)
            else:
                seq = (4,5,6)
            for k in seq:
                hqsom.update(test_data[k], g1, g2, s1, s2, a)
            flush(3)

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
    hqsom = HQSOM(9,65,3,use_pure_implementation=True)
    def flush(num):
        for l in range(num):
            hqsom.update(test_data[0], g1,g2,s1,s2,a)

    for j in range(40):
        for i in range(11):
            flush(3)
            seq = ()
            if i %2 == 0:
                seq = (1,2,3)
            else:
                seq = (4,5,6)
            for k in seq:
                hqsom.update(test_data[k], g1, g2, s1, s2, a)
            flush(3)

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
    
    blank_data = [Data_Image().data() for i in range(100)]
    #print len(square_data)
    #print len(diamond_data)
    #print len(x_data)

    #Paper settings
    #Make sure we don't use any of our "improvements"
    output_size = 17
    hqsom = PaperFig3Hierarchy(65,17,513,output_size, use_pure_implementation = True)
    g1,g2,g3,g4,s1,s2,s3,s4,a1,a2 = .1,.01,.1,.001, 16.0, 100.0, 4.0, 200.0, .1, .01
    run_name = "PAPER_"
    num_cycles, data_sets, num_repeats = 150, [("SQUARE",square_data), ("DIAMOND",diamond_data), ("X",x_data)], 5

    
    #Our settings
    #output_size = 4
    #hqsom = PaperFig3Hierarchy(20,10,25,output_size)
    #g1,g2,g3,g4,s1,s2,s3,s4,a1,a2 = .1,.08,.1,.008, 25.0, 80.0, 12.0, 75.0, .08, .005
    #run_name = "OUR_"
    #num_cycles, data_sets, num_repeats = 5, [("SQUARE",square_data), ("DIAMOND",diamond_data), ("X",x_data)], 1
    
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

    for data_name, data_collection in data_sets:
        mode = np.argmax(output_hash[data_name])
        num_items = float(len(data_collection))
        print "#"*80
        print "Data Set: {}".format(data_name)
        print "Most Frequently Classified As (MODE): {}".format(mode)
        results = np.array(output_hash[data_name])
        print "Full Distribution over Final RSOM Map Space:"
        print results / num_items
    MAP_Image(hqsom.bottom_hqsom_list[5].rsom.units,"output/FINAL_TOP_RSOM").save() 


if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "t:l", ["list","test="])
    except getopt.GetoptError, err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"

    #So that we get reproduceable results
    np.random.seed(15739)
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


