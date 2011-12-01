from som import *
from hqsom import *

tests = ("som","rsom", "hqsom", "hqsom_noise_multiple")

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

if __name__ == "__main__":
    for test in tests:
        print "#"*80
        print "Running test on: {}".format(test)
        print "-"*80
        try:
            eval("test_"+test)()
        except Exception as e :
            print e
            print "!!! ERROR !!!"
        else:
            print "SUCCESS"

        print "#"*80

