from som import *
from hqsom import *
from preproc.images import *
import getopt, sys
import traceback

tests = ("som","rsom", "hqsom", "hqsom_noise_multiple", "image_gen", "hqsom_77")

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
    
    g1,g2,s1,s2,a = .1,.1,16,100,.1
    hqsom = HQSOM(9,25,3)
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
    g1,g2,s1,s2,a = .1,.1,16,50,.1
    hqsom = HQSOM(9,25,3)
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
    for i in range(num_tests):
        try:
            test_hqsom_noise()    
        except:
            num_errors += 1
    print "Passed {} out of {}".format(num_tests-num_errors, num_tests)
    assert num_errors < .25 * num_tests

def test_image_gen():
    #SQUARE IMAGES
    Square_Image(7,(0,0)).save()
    #Generate 5x5
    start = [(0,0),(0,1),(0,2),
             (1,0),(1,1),(1,2),
             (2,0),(2,1),(2,2)]
    for s in start:
        Square_Image(5,s).save()
    #Generate 3x3
    for i in range(5):
        for j in range(5):
            Square_Image(3,(i,j)).save()

    #DIAMOND IMAGES
    Diamond_Image(3, (3,3)).save()
    #Generate 5x5
    for i in range(2,5):
        for j in range(2,5):
            Diamond_Image(2, (i,j)).save()
    #Generate 3x3
    for i in range(1,6):
        for j in range(1,6):
            Diamond_Image(1,(i,j)).save()
    #X IMAGES
    X_Image(7,(0,0)).save()
    for i in range(3):
        for j in range(3):
            X_Image(5,(i,j)).save()
    for i in range(5):
        for j in range(5):
            X_Image(3,(i,j)).save()

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
    #Generate the test sequence, note that we must to a spiral exposure to get the
    #correct temporal-spatial representations
    
    #7x7 only has one possible test
    coord_test = {"large":[(7,0,0)]}
    #5x5 has 9 possible positions
    coord_test["medium"] = [(5,i,j) for (i,j) in enumerate_spiral(2)]
    #3x3 has 25 possible positions
    coord_test["small"] = [(3,i,j) for (i,j) in enumerate_spiral(4)]
    #######The SQUARE data sets
    square_data = []
    #First we spiral out, then back in for each data set
    for data_set in ("large","medium", "small"):
        iteration = coord_test[data_set][::-1] + coord_test[data_set] 
        for (w,x,y) in iteration:
            image_data = Data_Image("data/square_{}_({}, {}).png".format(w,x,y))
            square_data.append(image_data.data())
    #for i in range(len(square_data)):
    #    im = square_data[i]
    #    im.save("square_test_{}.png".format(i), "PNG")
    #######
    #######The DIAMOND data sets
    diamond_data = []
    #First we spiral out, then back in for each data set
    for data_set in ("large","medium", "small"):
        iteration = coord_test[data_set][::-1] + coord_test[data_set] 
        for (w,x,y) in iteration:
            image_data = Data_Image("data/diamond_{}_({}, {}).png".format(w/2,x+w/2,y+w/2))
            diamond_data.append(image_data.data())
    #for i in range(len(diamond_data)):
    #    im = diamond_data[i]
    #    im.save("diamond_test_{}.png".format(i), "PNG")
    #######The X data sets
    x_data = []
    #First we spiral out, then back in for each data set
    for data_set in ("large","medium", "small"):
        iteration = coord_test[data_set][::-1] + coord_test[data_set] 
        for (w,x,y) in iteration:
            image_data = Data_Image("data/x_{}_({}, {}).png".format(w,x,y))
            x_data.append(image_data.data())
    #for i in range(len(x_data)):
    #    im = x_data[i]
    #    im.save("x_test_{}.png".format(i), "PNG")
    
    blank_data = [Data_Image().data() for i in range(100)]
    #print len(square_data)
    #print len(diamond_data)
    #print len(x_data)

    hqsom = PaperFig3Hierarchy(100,40,100,5)
    g1,g2,g3,g4,s1,s2,s3,s4,a1,a2 = .1,.01,.1,.001,16.0, 50.0, 4.0, 75.0, .1, .01
    seq_num = 0
    num_cycles, data_sets, num_repeats = 1, [square_data, diamond_data, x_data], 5
    total_run_count = num_cycles * len(data_sets)*(len(data_sets[0])*num_repeats+len(blank_data))
    for i in range(num_cycles):
        for data_set in data_sets:
            for j in range(num_repeats):
                for d in data_set:
                    hqsom.update(d,g1,g2,s1,s2,a1,g3,g4,s3,s4,a2)
                    print "update {}/{}".format(seq_num, total_run_count)
                    seq_num += 1
            for d in blank_data:
                hqsom.update(d,g1,g2,s1,s2,a1,g3,g4,s3,s4,a2)
                print "update {}/{}".format(seq_num, total_run_count)
                seq_num += 1

    print hqsom.activation_vector(blank_data[0])
    print hqsom.activation_vector(square_data[0])
    print hqsom.activation_vector(diamond_data[0])
    print hqsom.activation_vector(x_data[0])


if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "t:l", ["list","test="])
    except getopt.GetoptError, err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
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


